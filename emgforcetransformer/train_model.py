import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from datetime import datetime

# Training loop
def train_model(model, train_loader, val_loader,
                batches_before_validation=100,
                num_epochs=10, batch_size=32, learning_rate=1e-4):
    # Initialize wandb
    wandb.init(project="emgforcetransformer")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    global_step = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        for batch_idx, (emg_batch, force_batch) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            
            emg_batch = emg_batch.to(device)
            force_batch = force_batch.to(device)
            # Forward pass
            # [batch_size, num_chunks*chunk_secs*fps_emg, emg_channels] ->
            # [batch_size, num_chunks*chunk_secs*fps_force, force_channels]
            predicted_force = model(emg_batch)

            # Compute loss
            loss = criterion(predicted_force, force_batch)

            # Print training loss
            print(f'Training Loss at step {global_step}: {loss.item():.4f}')
            # Log training loss
            wandb.log({"Training Loss": loss.item(), "Global Step": global_step})

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
                    for val_idx, (emg_val, force_val) in enumerate(val_loader):
                        emg_val = emg_val.to(device)
                        force_val = force_val.to(device)

                        predicted_force_val = model(emg_val)
                        val_loss += criterion(predicted_force_val, force_val).item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

                print(f'Validation Loss at step {global_step}: {avg_val_loss:.4f}')
                wandb.log({"Validation Loss": avg_val_loss, "Global Step": global_step})

                model.train()  # Switch back to training mode

        # Optionally, you can save the model after each epoch
        # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print('Training complete.')
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model state dictionary with a timestamp in the filename
    model_filename = f'emg_force_transformer_{timestamp}.pth'
    torch.save(model.state_dict(), model_filename)
    wandb.finish()