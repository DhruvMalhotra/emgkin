from emgforcetransformer import EMGForceTransformer
from train_model import train_model
from load_data import create_dataloaders_lazy
import torch

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
cf = 80 # A Chunk's frames
fps = 2048 # Frames per second
t_sec = 25 # How long is a single file?

# transformer
d = 256
d_latent = 128
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 4

# Initialize lazy-loading DataLoaders
train_loader, val_loader = create_dataloaders_lazy(
    validation_fraction, bs,
    sc * cf, # frames in a sequence, sf
    t_sec * fps, # frames in file, ff
    r"C:\Users\Dhruv\Desktop\emgkin\data\1dof_dataset",
    20, 2, 5, 3)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
# Suppress specific UserWarning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# Initialize your model
model = EMGForceTransformer(device = device, d=d, d_latent=d_latent, channels_emg=channels_emg,
                            channels_force=channels_force,
                            bs = bs , sc = sc, cf = cf,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            nhead=nhead,
                            force_num_classes=force_num_classes, force_values_range=force_values_range)

# Start training
train_model(device, model, train_loader, val_loader,
            batches_before_validation=batches_before_validation,
            num_epochs=num_epochs,
            lr_max=lr_max)
