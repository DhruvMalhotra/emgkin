import torch
import torch.nn as nn
import math


class EMGForceTransformer(nn.Module):
    def __init__(self, d=512, d_latent=256, channels_emg=256, channels_force=5,
                 fps_emg=2048, fps_force=100,
                 chunk_secs=0.1, num_chunks=20,  # sequence length
                 num_encoder_layers=6, num_decoder_layers=6, nhead=8):
        super().__init__()
        self.d = d  # Embedding dimension
        self.d_latent = d_latent
        self.channels_emg = channels_emg
        self.channels_force = channels_force
        self.chunk_secs = chunk_secs  # chunk size in seconds
        self.num_chunks = num_chunks  # sequence length
        self.fps_emg = fps_emg
        self.fps_force = fps_force

        # Calculate frames in a chunk and ensure they are integers
        self.fc_emg = int(chunk_secs * fps_emg)  # frames in a chunk (emg)
        self.fc_force = int(chunk_secs * fps_force) # frames in a chunk (force)

        # Encoder Input: projection layer to map emg input to d
        self.input_projection = nn.Linear(self.fc_emg, d, bias=True)

        # Decoder Input: learnable rand. Called "object_queries" in https://arxiv.org/pdf/2005.12872 Fig 2
        self.object_queries = nn.Parameter(torch.randn(channels_force, d))

        # Decoder Output: projection layer to map transformer outputs to force data
        self.output_projection = nn.Linear(d, self.fc_force, bias=True)

        # Transformer model
        self.transformer = nn.Transformer(d_model=d, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=d_latent,
                                          batch_first=True)

        # Precompute positional embeddings
        # Shape: [num_chunks*channels_emg, d]
        self.emg_pos_embedding = self.get_emg_positional_embeddings()
        # Shape: [num_chunks*channels_force, d]
        self.force_pos_embedding = self.get_force_positional_embeddings()

    def get_emg_positional_embeddings(self):
        """
        Generates 4D positional embeddings for T and the EMG sensors arranged in 4 arrays of 8x8 each.
        """
        # Generate positions for each of the 256 EMG channels
        positions = []
        for chunk_idx in range(self.num_chunks):
            for array_idx in range(4):
                for x in range(8):
                    for y in range(8):
                        positions.append([chunk_idx, array_idx, x, y])
        positions = torch.tensor(
            positions, dtype=torch.float)  # Shape: [num_chunks*256, 4]

        # Compute sinusoidal positional embeddings
        pe = self.compute_positional_embeddings(positions, self.d)
        return pe  # Shape: [num_chunks*channels_emg, d]

    def get_force_positional_embeddings(self):
        """
        Generates 2D positional embeddings for T and the 5 force channels.
        """
        positions = []
        for chunk_idx in range(self.num_chunks):
            for f in range(self.channels_force):
                positions.append([chunk_idx, f])
        positions = torch.tensor(
            positions, dtype=torch.float)  # Shape: [num_chunks*5, 2]

        pe = self.compute_positional_embeddings(positions, self.d)
        return pe  # Shape: [num_chunks*channels_force, d]

    def compute_positional_embeddings(self, positions, d):
        """
        Computes sinusoidal positional embeddings for given positions.
        positions: Tensor of shape [num_positions, num_dims]
        d: Embedding dimension
        Returns: Tensor of shape [num_positions, d]
        """
        num_positions, num_dims = positions.shape
        pe = torch.zeros(num_positions, d)

        div_term = torch.exp(
            torch.arange(0, d // num_dims, 2) *
            (-math.log(10000.0) / (d // num_dims))
        )

        # For each dimension in positions, compute positional embeddings
        for dim in range(num_dims):
            pos = positions[:, dim].unsqueeze(1)  # Shape: [num_positions, 1]
            pe_dim = torch.zeros(num_positions, d // num_dims)
            pe_dim[:, 0::2] = torch.sin(pos * div_term)
            pe_dim[:, 1::2] = torch.cos(pos * div_term)
            pe[:, dim * (d // num_dims):(dim + 1) * (d // num_dims)] = pe_dim

        return pe

    def forward(self, emg_data, force_data):
        """
        emg_data: Tensor of shape [batch_size, num_chunks*chunk_secs*fps_emg, emg_channels]
        force_data: Tensor of shape [batch_size, num_chunks*chunk_secs*fps_force, force_channels]
        """

        batch_size = emg_data.shape[0]  # Extract batch size

        # Step 1: Assert shape
        assert emg_data.shape[1], self.num_chunks * self.chunk_secs * self.fps_emg
        assert force_data.shape[1], self.num_chunks * self.chunk_secs * self.fps_force

        # Step 2: Chunk the data into segments of chunk_secs
        # Reshape and transpose to get chunks: [batch_size, num_chunks, channels, frames in Chunk]
        emg_chunks = emg_data.view(
            batch_size, self.num_chunks, self.fc_emg, self.channels_emg).transpose(2, 3)
        force_chunks = force_data.view(
            batch_size, self.num_chunks, self.fc_force, self.channels_force).transpose(2, 3)

        # Step 3: Embed each emg chunk to D dimensions
        # [batch_size, num_chunks, channels_emg, d]
        v_emg_embedded = self.input_projection(emg_chunks) # nn.Linear applies to last dimension

        # Step 4: Repeat object queries, num_chunks time * batch_size times. Use .expand() to share same underlying data.
        v_force_embedded = self.object_queries.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.num_chunks, -1, -1)  # [num_chunks, channels_force, d]
        assert v_force_embedded.numel() == batch_size*self.num_chunks*self.channels_force*self.d

        # Step 5: Reshape to 1D sequence suitable for the transformer
        # Transformer expects input of shape [Batch Size, Sequence Length, Embedding Dim]
        emg_sequence = v_emg_embedded.reshape(
            batch_size, self.num_chunks*self.channels_emg, self.d)
        force_sequence = v_force_embedded.reshape(
            batch_size, self.num_chunks*self.channels_force, self.d)
        
        # Step 6: Add positional embeddings, unsqueeze to allow for batch dim
        emg_sequence += self.emg_pos_embedding.unsqueeze(0)
        force_sequence += self.force_pos_embedding.unsqueeze(0)


        # Step 7 & 8: Pass through transformer encoder and decoder
        transformer_output = self.transformer(
            src=emg_sequence, tgt=force_sequence)
        # transformer_output: [batch_size, num_chunks*self.channels_force, d]

        # Step 9: Reshape transformer output to [batch_size, num_chunks, channels_force, d]
        transformer_output = transformer_output.view(
            batch_size, self.num_chunks, self.channels_force, self.d)

        # Map transformer outputs to predicted force data
        # [batch_size, num_chunks, channels_force, fc_force]
        predicted_force = self.output_projection(transformer_output)  # nn.Linear applies to last dimension

        return predicted_force, force_chunks  # Return target for loss computation
