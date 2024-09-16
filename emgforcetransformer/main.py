import torch
import torch.nn.functional as F
from emgforcetransformer import EMGForceTransformer

# Sample training loop
if __name__ == "__main__":
    # Create the model
    # model = EMGForceTransformer()

    model = EMGForceTransformer(d=8, d_latent=5, channels_emg=256, channels_force=5,
                 fps_emg=2048, fps_force=100,
                 chunk_secs=0.1,
                 num_encoder_layers=1, num_decoder_layers=1, nhead=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    NUM_EPOCHS = 10  # Example number of epochs

    # Example input data (random tensors for illustration)
    # Replace with actual EMG and force data tensors
    emg_data = torch.randn(1, 204 * 20, 256)  # 2 seconds of EMG data
    force_data = torch.randn(1, 10 * 20, 5)    # 2 seconds of force data

    for epoch in range(NUM_EPOCHS):
        # Forward pass
        predicted_force, target_force = model(emg_data, force_data)

        # Compute loss
        loss = F.mse_loss(predicted_force, target_force)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")
