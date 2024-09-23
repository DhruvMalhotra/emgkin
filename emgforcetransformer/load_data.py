import wfdb
import torch
import os
import numpy as np
import random

class DataStreamer:
    def __init__(self, data_indices, file_identifiers,
                 bs, sf, ff,
                 rootpath):
        
        self.data_indices = data_indices  # Ordered subset of data indices
        self.file_identifiers = file_identifiers
        self.bs = bs  # Number of sequences in a batch.
        self.sf = sf  # Number of frames in each sequence.
        self.ff = ff  # Number of frames per file.
        self.rootpath = rootpath
        self.total_steps = (len(data_indices) * ff) // (sf * bs)

    def __iter__(self):
        """
        Iterator that yields batches of data in the form [bs, sf, num_channels].
        """
        batch_emg = []
        batch_force = []
        current_file_idx = None
        emg_data = None
        force_data = None

        for idx in self.data_indices:
            file_idx, seq_idx = idx

            if file_idx != current_file_idx:
                emg_data, force_data = self.load_file(file_idx)
                current_file_idx = file_idx

            # Extract the sequence from the loaded data
            start_idx = seq_idx * self.sf
            end_idx = start_idx + self.sf

            emg_sequence = emg_data[start_idx:end_idx]
            force_sequence = force_data[start_idx:end_idx]

            # Ensure sequence lengths are correct
            if emg_sequence.shape[0] != self.sf:
                continue  # Skip incomplete sequences at the end
            if force_sequence.shape[0] != self.sf:
                continue

            batch_emg.append(emg_sequence)
            batch_force.append(force_sequence)

            # Yield batch if batch size is reached
            if len(batch_emg) == self.bs:
                batch_emg_tensor = torch.stack(batch_emg)  # Shape: [bs, sf, num_emg_channels]
                batch_force_tensor = torch.stack(batch_force)  # Shape: [bs, sf, num_force_channels]
                yield batch_emg_tensor, batch_force_tensor
                batch_emg = []
                batch_force = []

        # Yield any remaining sequences in the batch
        if batch_emg:
            batch_emg_tensor = torch.stack(batch_emg)
            batch_force_tensor = torch.stack(batch_force)
            yield batch_emg_tensor, batch_force_tensor
    
    def load_file(self, file_idx):
         # Load the new file
        subject_idx, session_idx, finger_idx, sample_idx = self.file_identifiers[file_idx]

        suffix = f'finger{finger_idx+1}_sample{sample_idx+1}'
        prefix = f'subject{subject_idx+1:02}_session{session_idx+1}'

        emg_file_path = os.path.join(self.rootpath, prefix, f'1dof_preprocess_{suffix}')
        force_file_path = os.path.join(self.rootpath, prefix, f'1dof_force_{suffix}')

        # Load the entire file
        try:
            record_emg = wfdb.rdrecord(emg_file_path)
            emg_data_full = torch.tensor(record_emg.p_signal, dtype=torch.float32)
            assert emg_data_full.shape[0] == self.ff, "EMG frames per file mismatch."
        except Exception as e:
            raise IOError(f"Error loading EMG data from {emg_file_path}: {str(e)}")

        try:
            record_force = wfdb.rdrecord(force_file_path)
            force_data_full = torch.tensor(record_force.p_signal, dtype=torch.float32)
        except Exception as e:
            raise IOError(f"Error loading Force data from {force_file_path}: {str(e)}")

        # Upsample force data if necessary
        if force_data_full.shape[0] != emg_data_full.shape[0]:
            upsampled_force = upsample_fractional(force_data_full.T.numpy(),
                                                    emg_data_full.shape[0] / force_data_full.shape[0])
            force_data_full = torch.tensor(upsampled_force.T, dtype=torch.float32)

        return emg_data_full, force_data_full


def generate_data_indices(subjects, sessions, fingers, samples, frames_per_file, sequence_length):
    file_identifiers = []
    data_indices = []
    for subject_idx in range(subjects):
        for session_idx in range(sessions):
            for finger_idx in range(fingers):
                for sample_idx in range(samples):
                    file_identifiers.append((subject_idx, session_idx, finger_idx, sample_idx))

    num_sequences_per_file = frames_per_file // sequence_length

    for file_idx, _ in enumerate(file_identifiers):
        for seq_idx in range(num_sequences_per_file):
            data_indices.append((file_idx, seq_idx))

    return data_indices, file_identifiers

def split_data_indices(data_indices, validation_fraction):
    total_samples = len(data_indices)
    val_size = int(validation_fraction * total_samples)
    train_size = total_samples - val_size

    train_indices = data_indices[:train_size]
    val_indices = data_indices[train_size:]

    # Shuffle the train indices
    random.shuffle(train_indices)

    return train_indices, val_indices

def create_dataloaders_lazy(validation_fraction, bs, sf, ff, rootpath, subjects,
                          sessions=2, fingers=5, samples=3):
    """
    Create data streamers for training and validation.
    """
    data_indices, file_identifiers = generate_data_indices(
        subjects, sessions, fingers, samples, ff, sf)

    train_indices, val_indices = split_data_indices(data_indices, validation_fraction)

    # Sort train and validation indices to ensure sequential loading
    train_indices.sort()
    val_indices.sort()

    # Create data streamers
    train_streamer = DataStreamer(train_indices, file_identifiers, bs,
                                  sf, ff, rootpath)
    val_streamer = DataStreamer(val_indices, file_identifiers, bs,
                                sf, ff, rootpath)

    return train_streamer, val_streamer

def upsample_fractional(signal, factor):
    """
    Upsample a 2D signal by a given factor using linear interpolation.
    """
    num_channels, original_time = signal.shape
    new_time = int(np.round(original_time * factor))
    old_time_indices = np.arange(original_time)
    new_time_indices = np.linspace(0, original_time - 1, new_time)
    upsampled_signal = np.zeros((num_channels, new_time))

    for channel in range(num_channels):
        upsampled_signal[channel, :] = np.interp(
            new_time_indices, old_time_indices, signal[channel, :])

    return upsampled_signal
