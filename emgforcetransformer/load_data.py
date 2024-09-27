import wfdb
import torch
import os
import numpy as np
import random

class DataStreamer:
    def __init__(self, sequence_indices, file_identifiers,
                 bs, sf, ff,
                 rootpath):
        
        self.sequence_indices = sequence_indices  # Ordered set of Frames
        self.file_identifiers = file_identifiers
        self.bs = bs  # Number of sequences in a batch.
        self.sf = sf  # Number of frames in each sequence.
        self.ff = ff  # Number of frames per file.
        self.rootpath = rootpath
        self.total_steps = len(sequence_indices) / (bs)

    def __iter__(self):
        """
        Iterator that yields batches of data in the form [bs, sf, num_channels].
        """
        batch_emg = []
        batch_force = []
        current_file_idx = None
        emg_data = None
        force_data = None

        for idx in self.sequence_indices:
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
                                                    emg_data_full.shape[0] / force_data_full.shape[0]).T
            force_data_full = torch.tensor(upsampled_force, dtype=torch.float32)

        return emg_data_full, force_data_full


def generate_sequence_indices(subjects, sessions, fingers, samples, frames_per_file, sequence_length):
    file_identifiers = []
    sequence_indices = []
    for subject_idx in range(subjects):
        for session_idx in range(sessions):
            for finger_idx in range(fingers):
                for sample_idx in range(samples):
                    file_identifiers.append((subject_idx, session_idx, finger_idx, sample_idx))

    num_sequences_per_file = frames_per_file // sequence_length

    for file_idx, _ in enumerate(file_identifiers):
        for seq_idx in range(num_sequences_per_file):
            sequence_indices.append((file_idx, seq_idx))

    return sequence_indices, file_identifiers

def split_sequence_indices(data_indices, validation_fraction):
    total_samples = len(data_indices)
    val_size = int(validation_fraction * total_samples)
    train_size = total_samples - val_size

    train_indices = data_indices[:train_size]
    val_indices = data_indices[train_size:]

    return train_indices, val_indices

def create_dataloaders_lazy(validation_fraction, bs, sf, ff, rootpath, dataload):
    """
    Create data streamers for training and validation.
    """
    subjects = dataload[0]
    sessions = dataload[1]
    fingers = dataload[2]
    samples = dataload[3]

    sequence_indices, file_identifiers = generate_sequence_indices(
        subjects, sessions, fingers, samples, ff, sf)

    train_indices, val_indices = split_sequence_indices(sequence_indices, validation_fraction)

    # Create data streamers
    train_streamer = DataStreamer(train_indices, file_identifiers, bs,
                                  sf, ff, rootpath)
    val_streamer = DataStreamer(val_indices, file_identifiers, bs,
                                sf, ff, rootpath)

    assert (len(train_indices) + len(val_indices) == (subjects * sessions * fingers * samples * ff / sf)), (
        f"Assertion failed: The number of train and validation sequences ({len(train_indices) + len(val_indices)}) "
        f"does not match the expected value ({subjects * sessions * fingers * samples * ff / sf}). "
        f"Details: len(train_indices)={len(train_indices)}, len(val_indices)={len(val_indices)}, "
        f"subjects={subjects}, sessions={sessions}, fingers={fingers}, samples={samples}, ff={ff}.")
    
    print(f"Total Sequences, training: {len(train_indices)}, validation: {len(val_indices)}. Steps/Epoch: {train_streamer.total_steps}")
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
