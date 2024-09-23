from emgforcetransformer import EMGForceTransformer
import torch
import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from force_classify import collapse

# Define your parameters

# training loop
validation_fraction = 0.25
batches_before_validation = 10
lr_max = 1e-4
num_epochs = 5
force_num_classes = 10
force_values_range = (-0.2, 0.2)

# data
channels_emg = 256
channels_force = 5
bs = 8 # A Batch's sequences
sc = 16 # A Sequence's chunks
cf = 16 # A Chunk's frames
fps = 2048 # Frames per second
t_sec = 25 # How long is a single file?
assert bs*sc*cf == fps*25 # One file should be eaten together

# transformer
d = 256
d_latent = 128
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 4

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

# Initialize your model
model = EMGForceTransformer(device = device, d=d, d_latent=d_latent, channels_emg=channels_emg,
                            channels_force=channels_force,
                            bs = bs , sc = sc, cf = cf,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            nhead=nhead,
                            force_num_classes=force_num_classes, force_values_range=force_values_range)

model.load_state_dict(torch.load(
    os.path.join(script_dir,'model_saves',
                 'emg_force_transformer_20240922_105644.pth')))
model.eval()

predicted_force = model(emg_batch.unsqueeze(0)) # add the batch dimension.
assert predicted_force.shape[0] == 1
predicted_force = predicted_force.squeeze(0)

print("predicted_force_shape: " + str(predicted_force.shape))
print("predicted_force: " + str(predicted_force[0][0]))
# Move predictions to CPU and detach from computation graph
# Shape: [sc * cf, channels_force, num_classes]
predicted_force_classes = predicted_force.detach().cpu().squeeze(0)  

# Shape: [sc * cf, channels_force]
predicted_force_value = collapse(predicted_force, force_num_classes, force_values_range)

# Convert tensors to NumPy arrays for plotting
actual_force = force_batch.cpu().numpy()  # Shape: [num_samples, 5]
predicted_force = predicted_force_value.cpu().numpy()

# Create a time axis based on fps
num_samples = actual_force.shape[0]
time = np.arange(num_samples) / fps  # Time in seconds

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