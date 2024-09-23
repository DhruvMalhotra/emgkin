import torch
import torch.nn as nn
import math
from einops import rearrange

class EMGForceTransformer(nn.Module):
    def __init__(self, device, channels_emg, channels_force,
                 bs, sc, cf,
                 d, d_latent, num_encoder_layers, num_decoder_layers, nhead,
                 force_num_classes, force_values_range):  # Number of force classification buckets)
        super().__init__()
        self.d = d  # Embedding dimension
        self.d_latent = d_latent
        self.channels_emg = channels_emg
        self.channels_force = channels_force
        self.cf = cf  # a chunk has this many frames
        self.sc = sc  # a seqience has this many chunks
        self.bs = bs  # a batch has this many sequences
        self.force_num_classes = force_num_classes  # Number of classes (buckets)
        self.force_values_range = force_values_range # Force Range

        # Encoder Input: projection layer to map emg input to d
        self.input_projection = nn.Linear(self.cf, d, bias=True)

        # Decoder Input: learnable rand. Called "object_queries" in https://arxiv.org/pdf/2005.12872 Fig 2
        self.object_queries = nn.Parameter(torch.randn(channels_force, d))

        # Decoder Output: projection layer to map transformer outputs to logits over num_classes
        self.output_projection = nn.Linear(d, self.cf * self.force_num_classes, bias=True)
        #self.logits_projection = nn.Linear(self.fc, self.fc * self.num_classes, bias=True)

        # Transformer model
        self.transformer = nn.Transformer(d_model=d, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=d_latent,
                                          batch_first=True)

        # Precompute positional embeddings
        # Shape: [sc*channels_emg, d]
        self.emg_pos_embedding = self.get_emg_positional_embeddings().to(device)
        # Shape: [sc*channels_force, d]
        self.force_pos_embedding = self.get_force_positional_embeddings().to(device)

    def get_emg_positional_embeddings(self):
        """
        Generates 4D positional embeddings for T and the EMG sensors arranged in 4 arrays of 8x8 each.
        """
        # Generate positions for each of the 256 EMG channels
        positions = []
        for chunk_idx in range(self.sc):
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
        for chunk_idx in range(self.sc):
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
        # emg_data: Tensor of shape [bs, sc * cf, channels_emg]
        
        bs = emg_data.shape[0]  # Extract batch size
        assert emg_data.shape == (bs, self.sc * self.cf, self.channels_emg), \
            f"Expected emg_data.shape == {(bs, self.sc * self.cf, self.channels_emg)}, but got {emg_data.shape}"
        
        # Step 2.2: Chunk the data into sequences and batches
        # Reshape and transpose to get chunks: [bs, sc * channels_emg, cf]
        emg_chunks = rearrange(emg_data, 'a (b c) d -> a (b d) c', b=self.sc, c=self.cf)

        # Step 3: Embed each emg chunk to D dimensions
        # [bs, sc * channels_emg , d]
        v_emg_embedded = self.input_projection(emg_chunks) # nn.Linear applies to last dimension

        # Step 4: Prepare decoder input
        # Repeat object queries, bs * sc times.
        # object queries [channels_force, d]
        v_force_embedded = self.object_queries.unsqueeze(0).unsqueeze(0).expand(
            bs, self.sc, -1, -1)  # [bs, sc, channels_force, d]
        v_force_embedded = v_force_embedded.reshape(
            bs, self.sc*self.channels_force, self.d)
        assert v_force_embedded.shape == (bs, self.sc * self.channels_force, self.d), \
            f"Shape mismatch: Expected {(bs, self.sc * self.channels_force, self.d)}, " \
            f"but got {v_force_embedded.shape}"
        
        # Step 5: Add positional embeddings, unsqueeze to allow for batch dim
        # self.emg_pos_embedding shape is [sc * channels_emg, d]
        # v_emg_embedded.shape: [bs, sc * channels_emg, d]
        emg_sequence = v_emg_embedded + self.emg_pos_embedding[None, ...]
        force_sequence = v_force_embedded + self.force_pos_embedding[None, ...]

        
        
        # Step 6: Pass through transformer encoder and decoder
        # Transformer expects input of shape [Batch Size, Sequence Length, Embedding Dim]
        transformer_output = self.transformer(src=emg_sequence, tgt=force_sequence)
        # transformer_output: [bs, sc*self.channels_force, d]

        # Step 7.1: Reshape transformer output to [bs, sc, channels_force, d]
        transformer_output = rearrange(transformer_output,
                                       'a (b c) d -> a b c d',
                                       b=self.sc, c=self.channels_force)

        # Step 7.2: Apply the output learnable layer to get logits over chunks
        # [bs, sc, channels_force, cf * num_classes]
        predicted_force = self.output_projection(transformer_output)

        # Step 7.3: Prepare final output for loss
        # [bs, sc * cf , channels_force, num_classes]
        predicted_force = rearrange(predicted_force,
                                    'a b c (d e) -> a (b d) c e',
                                    d=self.cf, e=self.force_num_classes)
        
        # Return: [bs, sf, channels_force, num_classes]
        return predicted_force