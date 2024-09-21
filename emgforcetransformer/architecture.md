# EMG-Force Data Transformer Workflow

### **Step 1: Raw Data Shape**
- **EMG:**
  - 64 channels, 4 arrays → 256 channels total
  - `fps_emg = 2048`
  - Shape: `[t_secs * fps_emg , 256]`
  
- **Force:**
  - 5 channels
  - `fps_force = 100`
  - Shape: `[t_secs * fps_force , 5]`

---

### **Step 2: Chunk 0.1s Together**
- Number of frames in a chunk: `fc = 0.1 * fps`
- Number of chunks: `num_chunks = t_secs / 0.1`

- **EMG:**
  - Shape: `[t_secs / 0.1 , 256 , fc_emg]`
  - `fc_emg = 0.1 * fps_emg = 0.1 * 2048 = 204.8 ≈ 204`

- **Force:**
  - Shape: `[t_secs / 0.1 , 5 , fc_force]`
  - `fc_force = 0.1 * fps_force = 0.1 * 100 = 10`

---

### **Step 3: Embed a Chunk to D Dimensions**
Both EMG and force chunks are mapped to the same dimension **D**:

- **EMG:**
  - Shape: `[num_chunks , 256 , D]`

- **Force:**
  - Shape: `[num_chunks , 5 , D]`

For embedding a single element of size `f`, we learn weight matrices `W`:
  
- We need to learn a matrix `W[D, f]` such that:
  - `W[D, f] x v[f] = v'[D]`

To introduce non-linearity, we use two learned matrices:
  
- `W_2[D, 256] x ReLU(W_1[256, f] x v[f])`

This means we learn the following:

- **For EMG:**
  - `W_2_emg[D, 256] x ReLU(W_1_emg[256, fc_emg] x v[fc_emg])`

- **For Force:**
  - `W_2_force[D, 256] x ReLU(W_1_force[256, fc_force] x v[fc_force])`

---

### **Step 4: Add Sinusoidal Positional Embedding**
- **For EMG:**
  - Add 3D positional embeddings corresponding to `[num_chunks , 256]` of dimension **D**.
  - The 3 dimensions are **8x8x4**.

- **For Force:**
  - Add 1D positional embeddings corresponding to `[num_chunks , 5]` of dimension **D**.
  - The 1 dimension represents the **5 finger channels**.

- **Final Shapes:**
  - **EMG:** `v[num_chunks , 256 , D] + pos[num_chunks , 256 , D]`
  - **Force:** `v[num_chunks , 5 , D] + pos[num_chunks , 5 , D]`

---

### **Step 5: Reshape to 1D Sequence for Transformer**
To prepare the data for the transformer, reshape the embeddings:

- **EMG:** `[N_emg, D]`, where `N_emg = num_chunks * 256`
- **Force:** `[N_force, D]`, where `N_force = num_chunks * 5`

---

### **Step 6: Transformer Encoder**
- The encoder consists of **Mx** layers.
- It takes in `[N_emg, D]` and applies **self-attention** to produce `[N_emg, D]`.

---

### **Step 7: Transformer Decoder**
- The decoder consists of **Nx** layers.
- It takes in `[N_force, D]` and performs **interleaved self-attention** and **cross-attention** with the encoder output.

---

### **Step 8: Loss Calculation**
- Use **Mean Square Error (MSE)** as the loss function to compute the difference between predicted and actual force values.
- Optimize the model using **Stochastic Gradient Descent (SGD)**.
