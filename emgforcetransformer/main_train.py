from emgforcetransformer import EMGForceTransformer
from mlp import MLP

from train_model import train_model
from load_data import create_dataloaders_lazy
from hyperparameters import *

# Initialize lazy-loading DataLoaders
train_loader, val_loader = create_dataloaders_lazy(
    validation_fraction, bs,
    sc * cf, # frames in a sequence, sf
    t_sec * fps, # frames in file, ff
    data_dir,
    main_train_dataload)

import warnings
# Suppress specific UserWarning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# Initialize your model
model = EMGForceTransformer(device = device, d_model=d_model, d_latent=d_latent, channels_emg=channels_emg,
                            channels_force=channels_force,
                            bs = bs , sc = sc, cf = cf,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            nhead=nhead,
                            force_num_classes=force_num_classes)
'''
model = MLP(sequence_length=sc*cf, channels_emg=channels_emg, channels_force=channels_force, force_num_classes=force_num_classes,
            hidden_dims=[256, 256, 256, 256, 256])
'''
# Start training
train_model(device, model, wandb_project_name,
            force_num_classes, force_values_range,
            train_loader, val_loader,
            batches_before_validation=batches_before_validation,
            num_epochs=num_epochs,
            lr_cold_start=lr_cold_start, lr_max=lr_max)
