import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sequence_length, channels_emg, channels_force, force_num_classes,
                 hidden_dims
                 ):
        super(MLP, self).__init__()

        self.sequence_length = sequence_length
        self.channels_emg = channels_emg
        self.channels_force = channels_force
        self.force_num_classes = force_num_classes

        layers = []
        input_dim = sequence_length * channels_emg
        output_dim = sequence_length * channels_force * force_num_classes

        prev_dim = input_dim

        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # You can choose other activation functions if desired
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input

        y = self.model(x) # [batch_size, sequence_length * channels_force * num_classes]

        y = y.view(y.size(0), self.sequence_length, self.channels_force, self.force_num_classes)
        # Return: [bs, sequence_length, channels_force, num_classes]
        return y
