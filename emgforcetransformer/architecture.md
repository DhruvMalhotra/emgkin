# EMG-Force Data Transformer Workflow

### **Step 1: Raw Data Shape**
- **EMG:**
  - 64 channels, 4 arrays → 256 channels total
  - `fps = 2048`
  - Shape: `[t_secs * fps_emg , 256]`
  
- **Upsample Force:**
  - 5 channels
  - `fps_force = 100` -> `fps = 2048`
  - Shape: `[t_secs * fps_force , 5]`

Shape: `[25 * fps , 256]`
Shape: `[25 * fps , 5]`

### **Step 2.1: Make sequences and batches**
  - Batch -> Sequence -> Chunk -> Frame
  - cf = a chunk's frames (or frames in a chunk)
  - sc = a sequence's chunks
  - sf = a sequence's frames = sc * cf
  - bs = a batch's sequences

Shape: `[bs, sc * cf , 256]`
Shape: `[bs, sc * cf , 5]`

### **Step 2.2: EMG: Chunk cf frames Together** and transpose to get chunks

Shape: `[bs, sc, cf , 256]` -> Shape: `[bs, sc * 256, cf]`

### **Step 3: EMG: Embed a Chunk to d Dimensions**
Transpose cf with the channels and map to the same dimension **d**:
Use nn.Linear with bias
d > 4*cf (ideally)

Shape: `[bs, sc * 256, d]`

### **Step 4: Decoder Input**
Object Queries
`[5, d] -> tiled to [bs, sc * 5 , d]`

### **Step 5: Add Sinusoidal Positional Embedding**
- **For EMG:**
  - Add 4D positional embeddings corresponding to `[sc , 256]` of dimension **d**.
  - The 4 dimensions are **Tx4x8x8**.

- **For Force:**
  - Add 2D positional embeddings corresponding to `[sc , 5]` of dimension **d**.
  - The dimensions represent **Time** and **5 finger channels**.


### **Step 6: Transformer**
- The encoder consists of **self-attention**.
- The decoder consists of **interleaved self-attention** and **cross-attention** with the encoder output.

### **Step 7.1: Reshape Transformer output to take out d**
`[bs, sc * 5 , d] -> [bs, sc, 5 , d]` 

### **Step 7.2: Apply the output learnable layer to get logits over chunks**
`[bs, sc * 5 , d] -> [bs, sc, 5 , cf * num_classes]` 

### **Step 7.3: Prepare final output for loss**
Input was `[bs, sc * cf , 256]`
So output has to be `[bs, sc * cf , 5, num_classes]`
`[bs, sc, 5 , fc * num_classes] -> [bs, sc * cf , 5, num_classes]` 

### **Step 8: Loss Calculation**
- Use **Cross Entropy Loss** as the loss function to compute the difference between predicted and actual force values.
- Optimize the model using **Stochastic Gradient Descent (SGD)**.
