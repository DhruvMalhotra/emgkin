import torch
import torch.nn as nn
import wandb
from datetime import datetime
import math
import os
from force_classify import classify

def get_lr(step, total_steps, lr_max, warmup_steps):
    """
    Calculate the learning rate with warmup and cosine decay
    """
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        return lr_max * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

def discretize_and_take_loss(force_gt, predicted_force, num_classes, values_range, device, criterion):
    # predicted_force.shape = [batch_size, num_chunks*chunk_secs*fps_force, channels_force, num_classes]
    
    # Discretize force values into class labels
    force_gt_labels = classify(force_gt, num_classes, values_range).to(device)

    # Reshape predictions to [batch_size * num_frames * channels_force, num_classes]
    predicted_force = predicted_force.reshape(-1, num_classes)
    force_gt_labels = force_gt_labels.reshape(-1)  # Flatten labels to match predictions

    # Compute loss
    return criterion(predicted_force, force_gt_labels)

# Training loop
def train_model(device, model, force_num_classes, force_values_range,
                train_loader, val_loader,
                batches_before_validation=100,
                num_epochs=10, lr_max=1e-4):
    # Initialize wandb
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(project = "emgforcetransformer-einsir")
    model = model.to(device)

    # Define optimizer and loss function (CrossEntropy for classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
    criterion = nn.CrossEntropyLoss()

    # Calculate total steps
    total_steps = train_loader.total_steps * num_epochs
    warmup_steps = int(total_steps * 0.02)
    global_step = 0

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define the directory
    model_dir_path = f'model_saves/emg_force_transformer_{timestamp}'
    os.makedirs(model_dir_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        for batch_idx, (emg_batch, force_gt) in enumerate(train_loader):
            # Update learning rate
            lr = get_lr(global_step, total_steps, lr_max, warmup_steps)
            for param_group in optimizer.param_groups:
               param_group['lr'] = lr
            #lr = lr_max

            emg_batch = emg_batch.to(device)
            force_gt = force_gt.to(device)

            # Forward pass
            # [bs, sc*cf, emg_channels] ->
            # [bs, sc*cf, force_channels, num_classes]
            predicted_force = model(emg_batch)

            # Compute loss
            loss = discretize_and_take_loss(force_gt, predicted_force,
                                            force_num_classes, force_values_range, device, criterion)

            # Print and log training loss
            print(f'Training Loss at step {global_step}, batch {batch_idx}: {loss.item():.4f}')
            wandb.log({"Training Loss": loss.item(), "Learning Rate": lr, "Global Step": global_step})

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation check
            if global_step % batches_before_validation == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for val_idx, (emg_val, force_val) in enumerate(val_loader):
                        emg_val = emg_val.to(device)
                        force_gt_val = force_val.to(device)

                        predicted_force_val = model(emg_val)
                        val_loss += discretize_and_take_loss(force_gt_val, predicted_force_val,
                                            force_num_classes, force_values_range, device, criterion).item()

                        val_steps += 1

                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

                print(f'Validation Loss at step {global_step}: {avg_val_loss:.4f}')
                wandb.log({"Validation Loss": avg_val_loss, "Global Step": global_step})

                torch.save(model.state_dict(), f'{model_dir_path}/{global_step}.pth')

                model.train()  # Switch back to training mode

            global_step += 1
    
    print('Training complete.')

    torch.save(model.state_dict(), f'model_saves/emg_force_transformer_{timestamp}/final_model.pth')
    wandb.finish()
