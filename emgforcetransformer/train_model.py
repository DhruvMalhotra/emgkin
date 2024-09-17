import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Function to create DataLoaders for training and validation
def create_dataloaders(emg_data, force_data, validation_fraction, batch_size):
    dataset = TensorDataset(emg_data, force_data)
    total_samples = len(dataset)
    val_size = int(validation_fraction * total_samples)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader_full = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader_full

# Training loop
def train_model(model, emg_data, force_data, validation_fraction=0.2,
                batches_before_validation=100, fraction_of_validation_set_to_infer=0.5,
                num_epochs=10, batch_size=32, learning_rate=1e-4):
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    emg_data = emg_data.to(device)
    force_data = force_data.to(device)

    # Create DataLoaders
    train_loader, val_loader_full = create_dataloaders(emg_data, force_data,
                                                       validation_fraction, batch_size)

    # Calculate number of validation batches to use
    total_val_batches = len(val_loader_full)
    val_batches_to_use = int(fraction_of_validation_set_to_infer * total_val_batches)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    global_step = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        for batch_idx, (emg_batch, force_batch) in enumerate(train_loader):
            emg_batch = emg_batch.to(device)
            force_batch = force_batch.to(device)

            # Forward pass
            predicted_force, target_force = model(emg_batch, force_batch)

            # Compute loss
            loss = criterion(predicted_force, target_force)

            # Print training loss
            print(f'Training Loss at step {global_step}: {loss.item():.4f}')

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Validation check
            if global_step % batches_before_validation == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for val_idx, (emg_val, force_val) in enumerate(val_loader_full):
                        if val_idx >= val_batches_to_use:
                            break
                        emg_val = emg_val.to(device)
                        force_val = force_val.to(device)

                        predicted_force_val, target_force_val = model(emg_val, force_val)
                        val_loss += criterion(predicted_force_val, target_force_val).item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
                print(f'Validation Loss at step {global_step}: {avg_val_loss:.4f}')
                model.train()  # Switch back to training mode

        # Optionally, you can save the model after each epoch
        # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print('Training complete.')