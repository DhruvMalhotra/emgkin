Each recording lasts t_secs (O(s))
EMG: 64 channels. 4 arrays = 256 channels, 	fps_emg = 2048
Force: 5 channels,					fps_force = 100 

# Step 1: Raw data shape:
[t_secs*fps_emg , 	256]
[t_secs*fps_force , 	5]

# Step 2: Chunk 0.1s together. fc (frames in chunk) = 0.1*fps. num_chunks = t_secs / 0.1
[t_secs / 0.1,	256, 	fc_emg]
[t_secs / 0.1, 	5, 	fc_force]

# Step 3: Embed a chunk to D dimensions. Map both emg and f to the same dimension D?
[num_chunks , 	256, 	D]
[num_chunks ,		5,	D]

# How?
# for a single element of size f,  we need learn a W[D, f] such that: W [D, f] x v[f] = v’[D]
# To introduce a nonlinearity:
W_2 [D, 256] x RelU( W_1[256,f] x v[f] )
# Learn W_1_emg, W_2_emg, W_1_force, W_2_force

	W_2_emg [D, 256] x RelU( W_1_emg [256, fc_emg] x v[fc_emg] )
	W_2_force [D, 256] x RelU( W_1_force [256, fc_force] x v[fc_force] )
	

# Step 4: add sinusoidal positional embedding
# For emg: 3D embeddings corresponding to [T’, 256] of dimension D. The 3 dims are: 8x8x4
# For force: 1D embeddings corresponding to [T’, 5] of dimension D. The 1 dim is the 5 finger channels
v[num_chunks , 256, D] + 	pos[num_chunks , 256, D]
v[num_chunks , 5, D] + 	pos[num_chunks , 5, D]

# Step 5: reshape to 1D-sequence for transformer, where N = num_chunks * num_channels
[N_emg, D]
[N_force, D]

# Step 6: Encoder. Mx layers encode [N_emg, D] -> [N_emg, D], self attention

# Step 7: Decoder: Nx layers of [N_force, D], interleaved self attention and cross attention.

# Step 8: Calculate mean square error probability, and SGD