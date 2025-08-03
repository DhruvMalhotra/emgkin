import os
import torch
__all__ = ['validation_fraction', 'batches_before_validation',
            'lr_cold_start', 'lr_max', 'num_epochs',
            'force_num_classes', 'force_values_range',
            'channels_emg', 'channels_force', 'bs', 'sc', 'cf',
            'fps', 't_sec',
            'd_model', 'd_latent', 'num_encoder_layers', 'num_decoder_layers', 'nhead',
            'main_train_dataload', 'device', 'base_dir', 'data_dir',
            'wandb_project_name'
           ]

# Define your parameters

# training loop
validation_fraction = 0.01
batches_before_validation = 100
lr_cold_start = (1e-1, 0)
lr_max = 1e-4
num_epochs = 500
force_num_classes = 10
force_values_range = (-0.2, 0.2)

# data
channels_emg = 256
channels_force = 5
bs = 8 # A Batch's sequences
sc = 8 # A Sequence's chunks
cf = 8 # A Chunk's frames
fps = 2048 # Frames per second
t_sec = 25 # How long is a single file?

# transformer
d_model = 32
d_latent = 32
num_encoder_layers = 2
num_decoder_layers = 2
nhead = 1

#### For main_train
# subjects, sessions = 2, fingers = 5, samples = 3
main_train_dataload = [1, 1, 1, 1]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', '..', 'data', '1dof_dataset')

wandb_project_name = 'emgforcetransformer-vit-overfit-new-pe'