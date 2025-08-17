# inference.py - Run inference with our custom transformer model

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- Import your project modules ---
from transformer_model import EMGHandPoseTransformer
from train import EMGPreprocessor
from emg2pose.data import Emg2PoseSessionData
from emg2pose.utils import downsample

# Add UmeTrack path for visualization
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
umetrack_path = os.path.join(current_dir, 'emg2pose', 'emg2pose', 'UmeTrack')
sys.path.insert(0, umetrack_path)

def load_latest_model(checkpoint_dir=".", window_size=200, num_channels=16, pose_dim=20, device=None):
    """Load the latest trained checkpoint of the custom Transformer model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")

    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "emg_transformer_epoch_*.pth")))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found, train the model first.")
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt}")

    # Load checkpoint
    ckpt = torch.load(latest_ckpt, map_location=device)

    # Recreate model with same hyperparameters
    model = EMGHandPoseTransformer(
        num_channels=num_channels,
        window_size=window_size,
        pose_dim=pose_dim,
        embed_dim=128,
        num_heads=8,
        num_layers_temporal=4,
        num_layers_spatial=2,
        mlp_hidden_dim=256,
        dropout=0.1,
        max_len=window_size
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    # Restore normalization stats
    pose_mean = ckpt["pose_mean"]
    pose_std = ckpt["pose_std"]
    model.set_target_normalization(torch.from_numpy(pose_mean), torch.from_numpy(pose_std))

    model.eval()
    return model, device, pose_mean, pose_std


def run_inference_on_session(model, device, session_data, pose_mean, pose_std, window_size=200, step_size=10):
    """Run sliding-window inference on EMG session data and return full temporal predictions."""
    # Take a slice of EMG data
    start, stop = 0, 10000  # 5 sec at 2 kHz
    emg = session_data[start:stop]["emg"]  # (T, 16)

    # Preprocess EMG
    preprocessor = EMGPreprocessor()
    emg_filtered = preprocessor.filter(emg)

    # Normalize EMG (fit fresh scaler - ideally should reuse training scaler)
    scaler = StandardScaler().fit(emg_filtered[:5000])
    emg_scaled = scaler.transform(emg_filtered)

    # Initialize prediction array to store full temporal sequence
    total_length = len(emg_scaled)
    pose_dim = 20  # Should match your model's pose_dim
    predictions = np.zeros((total_length, pose_dim))
    prediction_counts = np.zeros(total_length)  # Track how many predictions each timestep has
    
    num_windows = (len(emg_scaled) - window_size) // step_size + 1
    print(f"Running inference on {num_windows} windows with step_size={step_size}...")
    
    with torch.no_grad():
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > len(emg_scaled):
                break
                
            window = emg_scaled[start_idx:end_idx]  # (window_size, 16)
            
            # Convert to tensor and add batch dimension
            x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)  # (1, window_size, 16)
            
            # Forward pass - model outputs normalized predictions
            preds_norm = model(x)  # (1, window_size, pose_dim)
            
            # Denormalize predictions
            preds_denorm = model.denormalize(preds_norm)  # (1, window_size, pose_dim)
            
            # Add predictions to the full sequence (averaging overlapping regions)
            window_preds = preds_denorm[0].cpu().numpy()  # (window_size, pose_dim)
            predictions[start_idx:end_idx] += window_preds
            prediction_counts[start_idx:end_idx] += 1

    # Average overlapping predictions
    mask = prediction_counts > 0
    predictions[mask] = predictions[mask] / prediction_counts[mask, np.newaxis]
    
    # For any timesteps without predictions, use nearest neighbor interpolation
    if not mask.all():
        for dim in range(pose_dim):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                predictions[~mask, dim] = np.interp(
                    np.where(~mask)[0], 
                    valid_indices, 
                    predictions[valid_indices, dim]
                )
    
    print(f"Generated {len(predictions)} pose predictions at 2kHz")
    return predictions, emg_scaled


def visualize_results(preds, emg_data):
    """Visualize predictions and EMG signals, like visualize_pose.py"""
    
    print(f"Predictions shape: {preds.shape}")  # Debug info
    
    # --- Hand Pose Animation ---
    print("Generating hand pose animation...")
    
    # Downsample predictions to 60Hz for smoother animation
    # Now we have full temporal resolution, so downsample from 2kHz to 60Hz
    preds_60hz = downsample(preds, native_fs=2000, target_fs=60)
    print(f"Downsampled predictions shape: {preds_60hz.shape}")
    
    try:
        # Try to import and use the visualization module
        import emg2pose.visualization as visualization
        fig = visualization.get_plotly_animation_for_joint_angles(preds_60hz, color="lightblue")
        fig.write_html("hand_pose_animation_transformer.html")
        print("Saved hand pose animation to hand_pose_animation_transformer.html")
    except ImportError as e:
        print(f"Could not import visualization module: {e}")
        print("Skipping 3D animation generation...")
    except Exception as e:
        print(f"Error generating animation: {e}")
        print("Skipping 3D animation generation...")

    # --- EMG Signal Plot ---
    print("Generating EMG signal plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(8, 2, figsize=(15, 20), sharex=True)
    axs = axs.flatten()
    time_vector = np.arange(emg_data.shape[0]) / 2000.0  # Convert to seconds

    for i in range(min(emg_data.shape[1], 16)):  # Handle case where we have fewer channels
        axs[i].plot(time_vector, emg_data[:, i])
        axs[i].set_title(f'EMG Channel {i+1}')
        axs[i].set_ylabel('Signal Amplitude')
        axs[i].set_xlabel('Time (s)')

    fig.text(0.5, 0.04, 'Time (seconds)', ha='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.suptitle('Processed EMG Signals', fontsize=16)
    plt.savefig("emg_signals_processed.png", dpi=300)
    print("Saved EMG signals plot to emg_signals_processed.png")
    plt.close()

    # --- Predicted Poses Plot ---
    print("Generating predicted poses plot...")
    plt.figure(figsize=(15, 10))
    time_pred = np.arange(len(preds_60hz)) / 60.0  # Time in seconds at 60Hz
    
    # Plot more joint angles in a better layout
    n_joints = min(preds_60hz.shape[1], 20)  # Plot up to 20 joint angles
    rows = 4
    cols = 5
    
    for i in range(n_joints):
        plt.subplot(rows, cols, i+1)
        plt.plot(time_pred, preds_60hz[:, i], linewidth=1.5)
        plt.title(f'Joint {i+1}', fontsize=10)
        plt.xlabel('Time (s)', fontsize=8)
        plt.ylabel('Angle (rad)', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("predicted_hand_poses.png", dpi=300, bbox_inches='tight')
    print("Saved predicted poses plot to predicted_hand_poses.png")
    plt.close()
    
    # --- Summary statistics ---
    print(f"\nSummary:")
    print(f"  EMG data: {emg_data.shape[0]} samples at 2kHz ({emg_data.shape[0]/2000:.1f}s)")
    print(f"  Predictions: {preds.shape[0]} samples at 2kHz ({preds.shape[0]/2000:.1f}s)")
    print(f"  Animation: {preds_60hz.shape[0]} frames at 60Hz ({preds_60hz.shape[0]/60:.1f}s)")
    print(f"  Joint dimensions: {preds.shape[1]}")


if __name__ == "__main__":
    # Load model
    model, device, pose_mean, pose_std = load_latest_model(checkpoint_dir=".")

    # Pick session
    data_dir = "emg2pose_data/emg2pose_dataset_mini_spatial_ai" # CHANGE THE ADDRESS HERE
    session_file = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))[0]
    print(f"Loading session: {session_file}")
    session_data = Emg2PoseSessionData(hdf5_path=session_file)

    # Run inference
    preds, emg = run_inference_on_session(model, device, session_data, pose_mean, pose_std)

    # Visualize
    visualize_results(preds, emg)
    print("\nInference complete → check 'hand_pose_animation_transformer.html' & PNG files")
