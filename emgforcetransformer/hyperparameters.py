import torch
__all__ = ['validation_fraction', 'batches_before_validation', 'lr_max', 'num_epochs',
           'force_num_classes', 'force_values_range',
           'channels_emg', 'channels_force', 'bs', 'sc', 'cf',
           'fps', 't_sec',
            'd', 'd_latent', 'num_encoder_layers', 'num_decoder_layers', 'nhead',
            'device',
            'main_train_dataload']

# Define your parameters

# training loop
validation_fraction = 0.01
batches_before_validation = 5000
lr_max = 1e-4
num_epochs = 500
force_num_classes = 10
force_values_range = (-0.2, 0.2)

# data
channels_emg = 256
channels_force = 5
bs = 16 # A Batch's sequences
sc = 16 # A Sequence's chunks
cf = 1 # A Chunk's frames
fps = 2048 # Frames per second
t_sec = 25 # How long is a single file?

# transformer
d = 64
d_latent = 32
num_encoder_layers = 2
num_decoder_layers = 2
nhead = 4

#### For main_train
# subjects, sessions = 2, fingers = 5, samples = 3
main_train_dataload = [1, 1, 1, 1]


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')