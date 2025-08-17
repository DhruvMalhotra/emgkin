import os
import sys

# ---- START OF FIX ----
# Add the UmeTrack submodule path to the system path.
# This helps Python find the 'lib' module it's looking for.
# We use abspath to make sure this works regardless of where you run the script from.
current_dir = os.path.dirname(os.path.abspath(__file__))
umetrick_path = os.path.join(current_dir, 'emg2pose', 'emg2pose', 'UmeTrack')
sys.path.insert(0, umetrick_path)
# ---- END OF FIX ----


import torch
import glob
import random
import matplotlib.pyplot as plt
from pathlib import Path

from emg2pose.data import Emg2PoseSessionData
from emg2pose.utils import generate_hydra_config_from_overrides, downsample
from emg2pose.lightning import Emg2PoseModule
import emg2pose.visualization as visualization

def load_model_and_data():
    """Loads the pre-trained model and a random data session."""
    
    # --- Load Model ---
    DATA_DOWNLOAD_DIR = Path.home()
    checkpoint_path = DATA_DOWNLOAD_DIR / "emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run the download script first.")

    config = generate_hydra_config_from_overrides(
        overrides=[
            "experiment=tracking_vemg2pose",
            f"checkpoint={checkpoint_path}"
        ]
    )
    
    model = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )
    
    # --- Load Data ---
    data_dir = "emg2pose_data/emg2pose_dataset_mini_spatial_ai"
    sessions = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    
    if not sessions:
        raise FileNotFoundError(f"No data files found in {data_dir}. Please run the download script first.")
        
    session_path = "emg2pose_data/emg2pose_dataset_mini_spatial_ai/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-10_left.hdf5"
    print(f"Loading data from: {session_path}")
    session_data = Emg2PoseSessionData(hdf5_path=session_path)
    
    return model, session_data

def run_inference(model, session_data):
    """Runs inference on a window of the session data."""
    start_idx = 0
    stop_idx = 10000  # 5 seconds of data at 2kHz

    session_window = session_data[start_idx:stop_idx]
    no_ik_failure_window = session_data.no_ik_failure[start_idx:stop_idx]

    batch = {
        "emg": torch.Tensor([session_window["emg"].T]),
        "joint_angles": torch.Tensor([session_window["joint_angles"].T]),
        "no_ik_failure": torch.Tensor([no_ik_failure_window]),
    }

    preds, _, _ = model.forward(batch)
    
    # Detach from graph and convert to numpy
    preds = preds[0].T.detach().numpy()
    emg_data = batch["emg"][0].T.detach().numpy()

    return preds, emg_data

def visualize_results(preds, emg_data):
    """Visualizes the predicted pose and EMG signals."""
    
    # --- Hand Pose Animation ---
    print("Generating hand pose animation...")
    preds_60hz = downsample(preds, native_fs=2000, target_fs=60)
    fig = visualization.get_plotly_animation_for_joint_angles(preds_60hz, color="lightblue")
    fig.write_html("hand_pose_animation.html")
    print("Saved hand pose animation to hand_pose_animation.html")

    # --- EMG Signal Plot ---
    print("Generating EMG signal plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(8, 2, figsize=(15, 20), sharex=True)
    axs = axs.flatten()
    time_vector = range(emg_data.shape[0])
    
    for i in range(emg_data.shape[1]):
        axs[i].plot(time_vector, emg_data[:, i])
        axs[i].set_title(f'EMG Channel {i+1}')
        axs[i].set_ylabel('Signal Amplitude')
    
    fig.text(0.5, 0.04, 'Time (samples)', ha='center', va='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.suptitle('Raw EMG Signals', fontsize=16)
    plt.savefig("emg_signals.png", dpi=300)
    print("Saved EMG signals plot to emg_signals.png")
    plt.close()


if __name__ == "__main__":
    model, session_data = load_model_and_data()
    predicted_angles, emg_signals = run_inference(model, session_data)
    visualize_results(predicted_angles, emg_signals)
    print("\nVisualization complete. Check the generated HTML and PNG files.")
