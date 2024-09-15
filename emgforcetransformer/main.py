import torch
import torch.nn.functional as F
from emgforcetransformer import EMGForceTransformer

# Sample training loop
if __name__ == "__main__":
    # Create the model
    model = EMGForceTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    NUM_EPOCHS = 10  # Example number of epochs

    # Example input data (random tensors for illustration)
    # Replace with actual EMG and force data tensors
    emg_data = torch.randn(2048 * 10, 256)  # 10 seconds of EMG data
    force_data = torch.randn(100 * 10, 5)    # 10 seconds of force data

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
