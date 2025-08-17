# WHERE EVER MENTIONED AS 'Add the appropriate path here' PLEASE CONSIDER CHANGING THE PATH

# train.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import tqdm

import sys
sys.path.append('/kaggle/input/cod891/COD891/emg2pose') # Add the appropriate path here

# --- Project imports ---
from emg2pose.data import Emg2PoseSessionData

# --- Preprocessing / Dataset classes ---
class EMGPreprocessor:
    def __init__(self, lowcut=20, highcut=450, fs=2000, order=4):
        self.fs = fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band', analog=False)

    def filter(self, data: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, data, axis=0)


class TransformerInputDataset(Dataset):
    def __init__(self, emg_data, pose_data, window_size=200, step_size=5, scaler=None, preprocessor=None):
        self.window_size = window_size
        self.step_size = step_size

        if preprocessor:
            emg_data = preprocessor.filter(emg_data)

        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.emg_data = self.scaler.fit_transform(emg_data)
        else:
            self.scaler = scaler
            self.emg_data = self.scaler.transform(emg_data)

        self.pose_data = pose_data
        self.num_windows = (len(self.emg_data) - window_size) // step_size

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.step_size
        end = start + self.window_size

        input_window = torch.tensor(self.emg_data[start:end], dtype=torch.float32)
        target_pose_seq = torch.tensor(self.pose_data[start:end], dtype=torch.float32)
        return input_window, target_pose_seq


# --- Welford streaming mean/std ---
def compute_pose_mean_std(session_files, sample_per_file=5000):
    count = 0
    mean = None
    M2 = None

    for p in tqdm.tqdm(session_files, desc="Computing pose stats"):
        s = Emg2PoseSessionData(hdf5_path=p)
        poses = s[:]['joint_angles']
        if len(poses) == 0:
            continue

        if sample_per_file and len(poses) > sample_per_file:
            idx = np.linspace(0, len(poses)-1, sample_per_file).astype(int)
            poses_sample = poses[idx]
        else:
            poses_sample = poses

        if mean is None:
            pose_dim = poses_sample.shape[1]
            mean = np.zeros(pose_dim, dtype=np.float64)
            M2 = np.zeros(pose_dim, dtype=np.float64)

        for vec in poses_sample:
            count += 1
            delta = vec - mean
            mean += delta / count
            delta2 = vec - mean
            M2 += delta * delta2

    var = M2 / (count - 1)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# --- Training ---
def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {DEVICE}")

    # Hyperparameters
    WINDOW_SIZE = 200
    STEP_SIZE = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    POSE_DIM = 20
    NUM_CHANNELS = 16

    DATA_DIR = "/kaggle/input/cod891/COD891/emg2pose_data/emg2pose_dataset_mini_spatial_ai" # Add the appropriate path here
    session_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not session_files:
        raise FileNotFoundError(f"No data files in {DATA_DIR}")

    preprocessor = EMGPreprocessor()

    # Fit EMG scaler
    print("Fitting EMG scaler...")
    sample_session = Emg2PoseSessionData(hdf5_path=session_files[0])
    sample_emg = sample_session[:10000]['emg']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(preprocessor.filter(sample_emg))

    # Pose mean/std
    if os.path.exists("pose_mean.npy") and os.path.exists("pose_std.npy"):
        pose_mean_np = np.load("pose_mean.npy")
        pose_std_np = np.load("pose_std.npy")
        print("Loaded saved pose stats.")
    else:
        print("Computing pose mean/std...")
        pose_mean_np, pose_std_np = compute_pose_mean_std(session_files)
        np.save("pose_mean.npy", pose_mean_np)
        np.save("pose_std.npy", pose_std_np)

    # Build datasets
    all_datasets = []
    for session_path in tqdm.tqdm(session_files, desc="Processing sessions"):
        s = Emg2PoseSessionData(hdf5_path=session_path)
        emg, pose = s[:]['emg'], s[:]['joint_angles']
        if len(emg) > WINDOW_SIZE:
            all_datasets.append(TransformerInputDataset(emg, pose,
                window_size=WINDOW_SIZE, step_size=STEP_SIZE, scaler=scaler, preprocessor=preprocessor))

    full_dataset = ConcatDataset(all_datasets)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = EMGHandPoseTransformer(
        num_channels=NUM_CHANNELS,
        window_size=WINDOW_SIZE,
        pose_dim=POSE_DIM,
        embed_dim=128,
        num_heads=8,
        num_layers_temporal=4,
        num_layers_spatial=2,
        mlp_hidden_dim=256,
        dropout=0.1,
        max_len=WINDOW_SIZE
    ).to(DEVICE)
    model.set_target_normalization(torch.from_numpy(pose_mean_np), torch.from_numpy(pose_std_np))
    model.target_mean = model.target_mean.to(DEVICE)
    model.target_std = model.target_std.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Add LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # minimize val_loss
        factor=0.1,        # reduce LR by 10x
        patience=3,        # wait 3 epochs of no improvement
        verbose=True       # log LR changes
    )

    # --- Resume training ---
    checkpoint_files = sorted(glob.glob("emg_transformer_epoch_*.pth"))
    start_epoch = 1
    if checkpoint_files:
        # if you want to force resume from your custom uploaded ckpt, overwrite latest_ckpt here
        latest_ckpt = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_ckpt}")
        
        # allow non-weight objects (safe if you trust the file)
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # restore normalization (if available in checkpoint)
        if 'pose_mean' in ckpt and 'pose_std' in ckpt:
            pose_mean_np, pose_std_np = ckpt['pose_mean'], ckpt['pose_std']
            model.set_target_normalization(torch.from_numpy(pose_mean_np),
                                           torch.from_numpy(pose_std_np))
            model.target_mean = model.target_mean.to(DEVICE)
            model.target_std = model.target_std.to(DEVICE)

        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from checkpoint {latest_ckpt} at epoch {start_epoch}")

    # --- Training loop ---
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds_norm = model(inputs)

            mean_dev = model.target_mean.view(1, 1, -1)
            std_dev = model.target_std.view(1, 1, -1)
            targets_norm = (targets - mean_dev) / (std_dev + 1e-8)

            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                preds_norm = model(inputs)
                mean_dev = model.target_mean.view(1, 1, -1)
                std_dev = model.target_std.view(1, 1, -1)
                targets_norm = (targets - mean_dev) / (std_dev + 1e-8)
                val_loss += criterion(preds_norm, targets_norm).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Step LR scheduler with validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        ckpt_name = f"emg_transformer_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'pose_mean': pose_mean_np,
            'pose_std': pose_std_np
        }, ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
