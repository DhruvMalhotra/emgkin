from emgforcetransformer import EMGForceTransformer
import torch
import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np

# Define your parameters

# training loop
validation_fraction = 0.25
batches_before_validation = 10

# epoch/batch
num_epochs = 1
batch_size = 5
lr_max = 1e-4
chunk_secs = 1
num_chunks = 25
assert num_chunks*chunk_secs == 25 # Each file is 25s, and is a sequence

# transformer
d = 512
d_latent = 256
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 8

# data
channels_emg = 256
channels_force = 5
fps_emg = 2048
fps_force = 100


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))  # This will get the current script's directory
base_dir = os.path.join(script_dir,'..', 'data', '1dof_dataset')

# Define subject, session, and sample identifiers
subject = 'subject02'
session = 'session1'
force_sample = '1dof_force_finger4_sample1'
emg_sample = '1dof_preprocess_finger4_sample1'

# Construct file paths
force_record_path = os.path.join(base_dir, f'{subject}_{session}', force_sample)
emg_record_path = os.path.join(base_dir, f'{subject}_{session}', emg_sample)

# Load the records using wfdb
record_force = wfdb.rdrecord(force_record_path)
record_emg = wfdb.rdrecord(emg_record_path)
emg_batch = torch.tensor(record_emg.p_signal, dtype=torch.float32)
force_batch = torch.tensor(record_force.p_signal, dtype=torch.float32)

assert (fps_emg*chunk_secs).is_integer()
assert (fps_force*chunk_secs).is_integer()

# Initialize your model
model = EMGForceTransformer(d=d, d_latent=d_latent, channels_emg=channels_emg,
                            channels_force=channels_force,
                            fps_emg=fps_emg, fps_force=fps_force,
                            chunk_secs=chunk_secs,
                            num_chunks=num_chunks,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            nhead=nhead)

model.load_state_dict(torch.load(
    os.path.join(script_dir,'model_saves',
                 'emg_force_transformer_20240921_194443.pth')))
model.eval()

predicted_force = model(emg_batch.unsqueeze(0)) # add the batch dimension.
assert predicted_force.shape[0] == 1
predicted_force = predicted_force.squeeze(0)

print("predicted_force_shape: " + str(predicted_force.shape))
print("predicted_force: " + str(predicted_force[0][0]))
# Move predictions to CPU and detach from computation graph
predicted_force = predicted_force.detach().cpu().squeeze(0)  # Shape: [num_chunks * fps_force, force_channels, num_classes]

# Take the most likely logit to get class predictions
predicted_classes = predicted_force.argmax(dim=-1)  # Shape: [num_samples, force_channels]
# Map the classes back to force values in the range [-1, 1]
num_classes = model.num_classes  # Get number of classes from the model
bucket_size = 2 / num_classes  # Since force range is from -1 to 1
# Calculate the center of each bucket
predicted_force_values = -1 + (predicted_classes + 0.5) * bucket_size  # Shape: [num_samples, force_channels]

# Convert tensors to NumPy arrays for plotting
actual_force = force_batch.cpu().numpy()  # Shape: [num_samples, 5]
predicted_force = predicted_force_values.cpu().numpy()

# Create a time axis based on fps_force
fps_force = model.fps_force  # 100 fps
num_samples = actual_force.shape[0]
time = np.arange(num_samples) / fps_force  # Time in seconds

# Plotting section
plt.figure(figsize=(15, 20))  # For subplots
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.plot(time, actual_force[:, i], label='Actual Force', color='blue')
    plt.plot(time, predicted_force[:, i], label='Predicted Force', color='red', linestyle='--')
    plt.title(f'Finger {i + 1} Force Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()