import wfdb
import torch

def load_raw_data(rootpath, subject, sessions, fingers, samples):
    raw_emg_list = []
    raw_force_list = []
    for subject_idx in range(subject):
        for session_idx in range(sessions):
            for finger_idx in range(fingers):
                for sample_idx in range(samples):
                    suffix = 'finger'+str(finger_idx+1)+'_sample'+str(sample_idx+1)
                    prefix = 'subject0'+ str(subject_idx+1) +'_session'+ str(session_idx+1)
                    record_emg = wfdb.rdrecord(
                        rootpath + '/'+ prefix +'/1dof_preprocess_' + suffix)
                    record_force = wfdb.rdrecord(
                        rootpath + '/'+ prefix +'/1dof_force_' + suffix)
                    emg_raw = torch.tensor(record_emg.__dict__[
                                        # [2048*25, 256]
                                        'p_signal'], dtype=torch.float32)
                    force_raw = torch.tensor(record_force.__dict__[
                                            # [100*25, 5]
                                            'p_signal'], dtype=torch.float32)

                    raw_emg_list.append(emg_raw)
                    raw_force_list.append(force_raw)

    return torch.cat(raw_emg_list, dim=0), torch.cat(raw_force_list, dim=0)
