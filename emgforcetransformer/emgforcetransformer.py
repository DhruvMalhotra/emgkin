import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EMGForceTransformer(nn.Module):
    def __init__(self, d=512, d_latent=256, channels_emg=256, channels_force=5,
                 fps_emg=2048, fps_force=100,
                 chunk_secs=0.1,
                 num_encoder_layers=6, num_decoder_layers=6, nhead=8):
        super().__init__()
        self.d = d  # Embedding dimension
        # Latent dimension used in the MLP before and within the transformer
        self.d_latent = d_latent
        self.channels_emg = channels_emg
        self.channels_force = channels_force
        self.chunk_secs = chunk_secs  # chunk size in seconds
        self.fps_emg = fps_emg
        self.fps_force = fps_force

        # Calculate frames in a chunk and ensure they are integers
        self.fc_emg = int(chunk_secs * fps_emg)  # frames in a chunk (emg)
        # frames in a chunk (force)
        self.fc_force = int(chunk_secs * fps_force)

        # Embedding weight matrices initialized randomly
        self.w1_emg = nn.Parameter(torch.randn(d_latent, self.fc_emg))
        self.w2_emg = nn.Parameter(torch.randn(d, self.d_latent))
        self.w1_force = nn.Parameter(torch.randn(d_latent, self.fc_force))
        self.w2_force = nn.Parameter(torch.randn(d, self.d_latent))

        # Transformer model
        self.transformer = nn.Transformer(d_model=d, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=d_latent)

        # Output projection layer to map transformer outputs to force data
        # Chat GPT wants to put this in, why???
        self.output_projection = nn.Linear(d, self.fc_force)

        # Precompute positional embeddings
        self.emg_pos_embedding = self.get_emg_positional_embeddings(d)
        self.force_pos_embedding = self.get_force_positional_embeddings(d)

    def get_emg_positional_embeddings(self, d):
        """
        Generates 3D positional embeddings for the EMG sensors arranged in 4 arrays of 8x8 each.
        """
        # Generate positions for each of the 256 EMG channels
        positions = []
        for array_idx in range(4):
            for x in range(8):
                for y in range(8):
                    positions.append([array_idx, x, y])
        positions = torch.tensor(
            positions, dtype=torch.float)  # Shape: [256, 3]

        # Compute sinusoidal positional embeddings
        pe = self.compute_positional_embeddings(positions, d)
        return pe  # Shape: [256, D]

    def get_force_positional_embeddings(self, d):
        """
        Generates 1D positional embeddings for the 5 force channels.
        """
        positions = torch.arange(self.channels_force,
                                 # Shape: [5, 1]
                                 dtype=torch.float).unsqueeze(1)
        pe = self.compute_positional_embeddings(positions, d)
        return pe  # Shape: [5, D]

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
        emg_data: Tensor of shape [num_samples_emg, emg_channels]
        force_data: Tensor of shape [num_samples_force, force_channels]
        """

        # Step 1: Assuming both start at the same time,
        # get the duration (in secs) of the shorter stream.
        t_secs = min(emg_data.shape[0]/self.fps_emg,
                     force_data.shape[0]/self.fps_force)

        # Step 2: Chunk the data into segments of chunk_secs
        # Calculate the number of chunks that will fit
        num_chunks = int(t_secs/self.chunk_secs)

        # Trim extra samples to make data divisible by chunk size
        emg_data = emg_data[:num_chunks * self.fc_emg, :]
        force_data = force_data[:num_chunks * self.fc_force, :]

        # Reshape and transpose to get chunks: [t_secs/chunk_size, channels, frames in Chunk]
        emg_chunks = emg_data.view(
            num_chunks, self.fc_emg, self.channels_emg).transpose(1, 2)
        force_chunks = force_data.view(
            num_chunks, self.fc_force, self.channels_force).transpose(1, 2)

        # Step 3: Embed each chunk to D dimensions using learned weights and ReLU activation
        # For EMG data
        v_emg_w1 = torch.einsum('lf,Tcf->Tcl', self.w1_emg, emg_chunks)
        v_emg_relu = F.relu(v_emg_w1)
        # [num_chunks, channels_emg, d]
        v_emg_embedded = torch.einsum('dl,Tcl->Tcd', self.w2_emg, v_emg_relu)

        # For force data
        v_force_w1 = torch.einsum('lf,Tcf->Tcl', self.w1_force, force_chunks)
        v_force_relu = F.relu(v_force_w1)
        # [num_chunks, channels_force, d]
        v_force_embedded = torch.einsum(
            'dl,Tcl->Tcd', self.w2_force, v_force_relu)

        # Step 4: Add positional embeddings
        # Since positions are constant for channels, we can expand them over time
        emg_pos_embedding = self.emg_pos_embedding  # Shape: [channels_emg, d]
        # Shape: [channels_force, d]
        force_pos_embedding = self.force_pos_embedding

        # Expand positional embeddings to match [num_chunks, channels, d]
        emg_pos_embedding = emg_pos_embedding.unsqueeze(0).expand(
            num_chunks, -1, -1)  # [num_chunks, channels_emg, d]
        force_pos_embedding = force_pos_embedding.unsqueeze(0).expand(
            num_chunks, -1, -1)  # [num_chunks, channels_force, d]

        # Add positional embeddings
        v_emg_embedded += emg_pos_embedding
        v_force_embedded += force_pos_embedding

        # Step 5: Reshape to 1D sequence suitable for the transformer
        # Transformer expects input of shape [Sequence Length, Batch Size, Embedding Dim]
        # using batch size = 1 for now.
        emg_sequence = v_emg_embedded.view(
            num_chunks*self.channels_emg, 1, self.d)
        force_sequence = v_force_embedded.view(
            num_chunks*self.channels_force, 1, self.d)

        # Step 6 & 7: Pass through transformer encoder and decoder
        transformer_output = self.transformer(
            src=emg_sequence, tgt=force_sequence)
        # transformer_output: [num_chunks*self.channels_force, 1, d]

        # Reshape transformer output to [num_chunks, channels_force, d]
        transformer_output = transformer_output.view(
            num_chunks, self.channels_force, self.d)

        # Map transformer outputs to predicted force data
        # TODO(dhruv): Why is this needed?
        # [num_chunks, channels_force, fc_force]
        predicted_force = self.output_projection(transformer_output)

        # TODO(dhruv): Do we calculate the error between output and the non-Positionally encoded force chunks?
        return predicted_force, force_chunks  # Return target for loss computation
