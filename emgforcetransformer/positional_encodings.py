import math
import torch
import matplotlib.pyplot as plt

def generate_sinusoidal_positional_encoding(dims, d_model):
    """
    Generates a multidimensional sinusoidal positional encoding.

    Args:
        dims (list): A list of dimension sizes, e.g., [d1, d2, d3, d4].
        d_model (int): The total embedding dimension.

    Returns:
        Tensor of shape [total_positions, d_model], where total_positions = d1 * d2 * ... * dn
    """
    num_dims = len(dims)
    d_model_per_dim = d_model // num_dims
    assert d_model_per_dim % 2 == 0, "d_model divided by number of dimensions must be even"

    # Create positional encodings for each dimension
    pe_list = []
    for dim_size in dims:
        pe_dim = torch.zeros(dim_size, d_model_per_dim)
        position = torch.arange(0, dim_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model_per_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model_per_dim)
        )
        pe_dim[:, 0::2] = torch.sin(position * div_term)
        pe_dim[:, 1::2] = torch.cos(position * div_term)
        pe_list.append(pe_dim)  # Shape: [dim_size, d_model_per_dim]

    # Generate grid of positions
    meshgrids = torch.meshgrid([torch.arange(size) for size in dims], indexing='ij')
    # Flatten the grid to get positions of shape [total_positions, num_dims]
    positions_flat = torch.stack([g.flatten() for g in meshgrids], dim=-1)  # Shape: [total_positions, num_dims]

    # For each dimension, get the positional encoding for the positions along that dimension
    pe_per_dim = []
    for dim in range(num_dims):
        pos_indices = positions_flat[:, dim].long()  # Shape: [total_positions]
        pe_dim = pe_list[dim][pos_indices]           # Shape: [total_positions, d_model_per_dim]
        pe_per_dim.append(pe_dim)

    # Concatenate the positional encodings from each dimension
    pe = torch.cat(pe_per_dim, dim=-1)  # Shape: [total_positions, d_model]

    return pe

if __name__ == "__main__":
    
    # Test code
    d = 16
    num_positions = 100

    # Compute positional embeddings
    pe = generate_sinusoidal_positional_encoding([32], d)

    # Convert embeddings to NumPy array for plotting
    pe_np = pe.numpy()

    # Plot the embeddings as a 2D heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(pe_np, interpolation='nearest', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Positional Embeddings Heatmap')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position Index')
    plt.show()