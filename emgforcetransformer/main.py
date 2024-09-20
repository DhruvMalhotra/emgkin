from emgforcetransformer import EMGForceTransformer
from train_model import train_model
from load_data import load_raw_data

# Define your parameters

# training loop
validation_fraction = 0.25
batches_before_validation = 10
fraction_of_validation_set_to_infer = 0.5

# epoch/batch
num_epochs = 1
batch_size = 5
learning_rate = 1e-3
chunk_secs = 0.25
num_chunks = 20

# transformer
d = 8
d_latent = 5
num_encoder_layers = 2
num_decoder_layers = 2
nhead = 2

# data
channels_emg = 256
channels_force = 5
fps_emg = 2048
fps_force = 100
emg_raw, force_raw = load_raw_data(r"C:\Users\Dhruv\Desktop\emgkin\data\1dof_dataset", 4, 1, 5, 3)

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
emg_data = emg_raw.view(-1, num_chunks*int(fps_emg*chunk_secs), channels_emg)
force_data = force_raw.view(-1, num_chunks*int(fps_force*chunk_secs), channels_force)

print("emg_data.shape:"+str(emg_data.shape))
print("force_data.shape:"+str(force_data.shape))

# Start training
# Data has to be: [num_batches*batch_size, num_chunks*chunk_secs*fps_emg, emg_channels]
train_model(model, emg_data, force_data,
            validation_fraction=validation_fraction,
            batches_before_validation=batches_before_validation,
            fraction_of_validation_set_to_infer=fraction_of_validation_set_to_infer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate)
