from emgforcetransformer import EMGForceTransformer
from train_model import train_model
from load_data import create_dataloaders_lazy, debug_dataloader
import torch 

# Define your parameters

# training loop
validation_fraction = 0.25
batches_before_validation = 10

# epoch/batch
num_epochs = 5
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
# validation_set = load_raw_data()
# Initialize lazy-loading DataLoaders
train_loader, val_loader = create_dataloaders_lazy(
    validation_fraction, batch_size,
    r"C:\Users\Dhruv\Desktop\emgkin\data\1dof_dataset",
    20, 2, 5, 3)

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Start training
# Data has to be: [num_batches*batch_size, num_chunks*chunk_secs*fps_emg, emg_channels]
train_model(device, model, train_loader, val_loader,
            batches_before_validation=batches_before_validation,
            num_epochs=num_epochs,
            lr_max=lr_max)
