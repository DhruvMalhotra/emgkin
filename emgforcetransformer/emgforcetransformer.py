import torch
import torch.nn as nn
import math
from einops import rearrange

class EMGForceTransformer(nn.Module):
    def __init__(self, channels_emg, channels_force,
                 fps_emg, fps_force,
                 chunk_secs, num_chunks,  # sequence length
                 d=512, d_latent=256, num_encoder_layers=6, num_decoder_layers=6, nhead=8,
                 num_classes=100):  # Number of force classification buckets)
        super().__init__()
        self.d = d  # Embedding dimension
        self.d_latent = d_latent
        self.channels_emg = channels_emg
        self.channels_force = channels_force
        self.chunk_secs = chunk_secs  # chunk size in seconds
        self.num_chunks = num_chunks  # sequence length
        self.fps_emg = fps_emg
        self.fps_force = fps_force
        self.num_classes = num_classes  # Number of classes (buckets)

        # Calculate frames in a chunk and ensure they are integers
        self.fc_emg = int(chunk_secs * fps_emg)  # frames in a chunk (emg)
        self.fc_force = int(chunk_secs * fps_force) # frames in a chunk (force)

        # Encoder Input: projection layer to map emg input to d
        self.input_projection = nn.Linear(self.fc_emg, d, bias=True)

        # Decoder Input: learnable rand. Called "object_queries" in https://arxiv.org/pdf/2005.12872 Fig 2
        self.object_queries = nn.Parameter(torch.randn(channels_force, d))

        # Decoder Output: projection layer to map transformer outputs to logits over num_classes
        self.output_projection = nn.Linear(d, self.fc_force * self.num_classes, bias=True)
        #self.logits_projection = nn.Linear(self.fc_force, self.fc_force * self.num_classes, bias=True)

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

    def forward(self, emg_data):
        """
        emg_data: Tensor of shape [batch_size, num_chunks*chunk_secs*fps_emg, channels_emg]
        """
        batch_size = emg_data.shape[0]  # Extract batch size

        # Step 1: Assert shape
        assert emg_data.shape[1] == self.num_chunks * self.chunk_secs * self.fps_emg, \
            f"Expected emg_data.shape[1] == {self.num_chunks * self.chunk_secs * self.fps_emg}, but got {emg_data.shape[1]}"

        # Step 2: Chunk the data into segments of chunk_secs
        # Reshape and transpose to get chunks: [batch_size, num_chunks, channels, frames in Chunk]
        # emg_chunks = emg_data.view(
        #    batch_size, self.num_chunks, chunk_secs*fps_emg, self.channels_emg).transpose(2, 3)
        emg_chunks = rearrange(emg_data, 'a (b c) d -> a (b d) c', b=self.num_chunks, c=self.fc_emg)

        # Step 3: Embed each emg chunk to D dimensions
        # [batch_size, num_chunks*channels_emg, d]
        v_emg_embedded = self.input_projection(emg_chunks) # nn.Linear applies to last dimension

        # Step 4: Prepare decoder input
        # Repeat object queries, num_chunks * batch_size times. Use .expand() to share same underlying data.
        # object queries [channels_force, d]
        v_force_embedded = self.object_queries.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.num_chunks, -1, -1)  # [batch_size, num_chunks, channels_force, d]
        v_force_embedded = v_force_embedded.reshape(
            batch_size, self.num_chunks*self.channels_force, self.d)
        assert v_force_embedded.shape == (batch_size, self.num_chunks * self.channels_force, self.d), \
            f"Shape mismatch: Expected {(batch_size, self.num_chunks * self.channels_force, self.d)}, " \
            f"but got {v_force_embedded.shape}"

        # Step 6: Add positional embeddings, unsqueeze to allow for batch dim
        # self.emg_pos_embedding shape is [num_chunks*channels_emg, d]
        # v_emg_embedded.shape: [batch_size, num_chunks*channels_emg, d]
        device = emg_data.device
        emg_pos_embedding = self.emg_pos_embedding.to(device)
        force_pos_embedding = self.force_pos_embedding.to(device)
        emg_sequence = v_emg_embedded + emg_pos_embedding[None, ...]
        force_sequence = v_force_embedded + force_pos_embedding[None, ...]

        # Step 7 & 8: Pass through transformer encoder and decoder
        # Transformer expects input of shape [Batch Size, Sequence Length, Embedding Dim]
        transformer_output = self.transformer(
            src=emg_sequence, tgt=force_sequence)
        # transformer_output: [batch_size, num_chunks*self.channels_force, d]

        # Step 9: Reshape transformer output to [batch_size, num_chunks, channels_force, d]
        transformer_output = rearrange(transformer_output,
                                       'a (b c) d -> a b c d',
                                       b=self.num_chunks, c=self.channels_force)

        # Map transformer outputs to predicted force data with logits over classes
        # [batch_size, num_chunks, channels_force, fc_force*num_classes]
        predicted_force = self.output_projection(transformer_output)
        predicted_force = rearrange(predicted_force,
                                    'a b c (d e) -> a (b d) c e',
                                    d=self.fc_force, e=self.num_classes)
        
        # Return: [batch_size, num_frames, channels_force, num_classes]
        return predicted_force