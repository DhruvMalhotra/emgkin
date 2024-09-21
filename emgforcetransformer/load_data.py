import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
import os

class EMGForceDataset(Dataset):
    def __init__(self, rootpath, subjects, sessions, fingers, samples):
        """
        Initialize the dataset with the required parameters.
        Instead of loading all data into memory, this will only keep track of file paths and indexing logic.
        """
        self.rootpath = rootpath
        self.subjects = subjects
        self.sessions = sessions
        self.fingers = fingers
        self.samples = samples
        self.data_indices = self._create_data_indices()

    def _create_data_indices(self):
        """
        Create a list of file indices to track all data points.
        This list will hold tuples indicating subject, session, finger, and sample for each file.
        """
        data_indices = []
        for subject_idx in range(self.subjects):
            for session_idx in range(self.sessions):
                for finger_idx in range(self.fingers):
                    for sample_idx in range(self.samples):
                        data_indices.append((subject_idx, session_idx, finger_idx, sample_idx))
        return data_indices

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data_indices)

    def __getitem__(self, idx):
        """
        Load a single data point (EMG and Force) given an index.
        The data is loaded dynamically when this method is called.
        """
        subject_idx, session_idx, finger_idx, sample_idx = self.data_indices[idx]
        suffix = f'finger{finger_idx+1}_sample{sample_idx+1}'
        prefix = f'subject{subject_idx+1:02}_session{session_idx+1}'

        # Construct the file path without extension
        emg_file_path = os.path.join(self.rootpath, prefix, f'1dof_preprocess_{suffix}')
        force_file_path = os.path.join(self.rootpath, prefix, f'1dof_force_{suffix}')

        # Check if .hea files exist
        if not os.path.exists(f"{emg_file_path}.hea"):
            raise FileNotFoundError(f"EMG header file not found: {emg_file_path}.hea")
        if not os.path.exists(f"{force_file_path}.hea"):
            raise FileNotFoundError(f"Force header file not found: {force_file_path}.hea")

        # Load the EMG data for this index
        try:
            record_emg = wfdb.rdrecord(emg_file_path)
            emg_raw = torch.tensor(record_emg.p_signal, dtype=torch.float32)
        except Exception as e:
            raise IOError(f"Error loading EMG data from {emg_file_path}: {str(e)}")

        # Load the Force data for this index
        try:
            record_force = wfdb.rdrecord(force_file_path)
            force_raw = torch.tensor(record_force.p_signal, dtype=torch.float32)
        except Exception as e:
            raise IOError(f"Error loading Force data from {force_file_path}: {str(e)}")

        return emg_raw, force_raw

# Update the DataLoader creation to use batch loading
def create_dataloaders_lazy(validation_fraction, batch_size, rootpath, subject, sessions=2, fingers=5, samples=3):
    """
    Create data loaders for training and validation.
    Only loads batches as needed to avoid memory issues.
    """
    dataset = EMGForceDataset(rootpath, subject, sessions, fingers, samples)

    # Split into train and validation sets
    total_samples = len(dataset)
    val_size = int(validation_fraction * total_samples)
    train_size = total_samples - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders that load data batch by batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Debugging function
def debug_dataloader(dataloader, num_batches=5):
    unique_files = set()
    for batch_idx, (emg_batch, force_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"EMG shape: {emg_batch.shape}")
        print(f"Force shape: {force_batch.shape}")
        if batch_idx >= num_batches - 1:
            break
    print(f"Total unique files loaded: {len(unique_files)}")
    print(f"Sample of loaded files: {list(unique_files)[:5]}")
