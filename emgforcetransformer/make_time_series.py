from emgforcetransformer import EMGForceTransformer
import torch
import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from force_classify import collapse
from einops import rearrange
from load_data import upsample_fractional

# Define your parameters

# training loop
validation_fraction = 0.005
batches_before_validation = 50
lr_max = 1e-4
num_epochs = 5
force_num_classes = 10
force_values_range = (-0.2, 0.2)

# data
channels_emg = 256
channels_force = 5
bs = 5 # A Batch's sequences
sc = 32 # A Sequence's chunks
cf = 160 # A Chunk's frames
fps = 2048 # Frames per second
t_sec = 25 # How long is a single file?

# transformer
d = 512
d_latent = 128
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 4

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))  # This will get the current script's directory
base_dir = os.path.join(script_dir,'..', 'data', '1dof_dataset')

import warnings
# Suppress specific UserWarning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

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
emg_batch = torch.tensor(record_emg.p_signal, dtype=torch.float32).to(device)
force_gt = torch.tensor(record_force.p_signal, dtype=torch.float32).to(device)

#upsample force
force_gt = upsample_fractional(force_gt.cpu().numpy().T, emg_batch.shape[0] / force_gt.shape[0]).T

# Initialize your model
model = EMGForceTransformer(device = device, d=d, d_latent=d_latent, channels_emg=channels_emg,
                            channels_force=channels_force,
                            bs = bs , sc = sc, cf = cf,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            nhead=nhead,
                            force_num_classes=force_num_classes, force_values_range=force_values_range).to(device)

model.load_state_dict(torch.load(
    os.path.join(script_dir,'model_saves', 'emg_force_transformer_20240923_034302', 
                 '100.pth')))
model.eval()

# Initialize a list to store predictions
predicted_force_list = []

# emg_batch is Tensor of shape [ff, channels_emg]
# Model needs Tensor of shape [bs, sc * cf, channels_emg]
emg_batchs_and_sequences = rearrange(emg_batch, '(a b) c -> a b c', b=sc*cf)

print("emg_batch shape: " + str(emg_batch.shape))
print("emg_batchs_and_sequences: " + str(emg_batchs_and_sequences.shape))

# Process the data in batches sequentially
for i in range(0, emg_batchs_and_sequences.size(0), bs):
    batch_emg = emg_batchs_and_sequences[i:i+bs]
    predicted_batch_force = model(batch_emg)
    print("batch_emg " + str(batch_emg.shape))
    print("predicted_batch_force shape: " + str(predicted_batch_force.shape))

    # Model outputs Tensor of shape [bs, sc * cf, channels_force, num_classes]
    predicted_batch_force = rearrange(predicted_batch_force, 'a b c d -> (a b) c d')

    # Append the predicted forces to the list
    predicted_force_list.append(predicted_batch_force)

# Concatenate the predictions from all batches along the first dimension
predicted_force_classes = torch.cat(predicted_force_list, dim=0)

print("predicted_batch_force shape: " + str(predicted_batch_force.shape))
print("predicted_force_classes shape: " + str(predicted_force_classes.shape))

# Shape: [bs * sc * cf, channels_force]
predicted_force_value = collapse(predicted_force_classes, force_num_classes, force_values_range)

# Convert tensors to NumPy arrays for plotting
predicted_force = predicted_force_value.cpu().numpy()

print("predicted_force_value shape: " + str(predicted_force_value.shape))
print("force_gt shape: " + str(force_gt.shape))

# Downsample
force_gt_downsampled = upsample_fractional(force_gt.T, 0.01).T
predicted_force_downsampled = upsample_fractional(predicted_force.T, 0.01).T

# Create a time axis based on fps
num_samples = force_gt_downsampled.shape[0]
time = np.arange(num_samples) / fps  # Time in seconds

# Plotting section
plt.figure(figsize=(15, 20))  # For subplots
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.plot(time, force_gt_downsampled[:, i], label='Actual Force', color='blue')
    plt.plot(time, predicted_force_downsampled[:, i], label='Predicted Force', color='red', linestyle='--')
    plt.title(f'Finger {i + 1} Force Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()