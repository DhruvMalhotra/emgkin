from emgforcetransformer import EMGForceTransformer
from train_model import train_model
import torch

# Example usage:
# Assuming you have emg_data_tensor and force_data_tensor as your datasets
# emg_data_tensor: [total_samples, frames_emg, emg_channels]
# force_data_tensor: [total_samples, frames_force, force_channels]

# Initialize your model
model = EMGForceTransformer(d=8, d_latent=5, channels_emg=256, channels_force=5,
                            fps_emg=2050, fps_force=100,
                            chunk_secs=0.1,
                            num_encoder_layers=1, num_decoder_layers=1, nhead=1)
 
# Define your parameters
validation_fraction = 0.2
batches_before_validation = 13
fraction_of_validation_set_to_infer = 0.5
num_epochs = 10
batch_size = 2
learning_rate = 1e-4

emg_data = torch.randn(100, 2050*2, 256)
force_data = torch.randn(100, 100*2, 5)

# Start training
train_model(model, emg_data, force_data,
            validation_fraction=validation_fraction,
            batches_before_validation=batches_before_validation,
            fraction_of_validation_set_to_infer=fraction_of_validation_set_to_infer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate)
