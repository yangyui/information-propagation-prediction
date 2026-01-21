import torch
import math


class TimeEncoding(torch.nn.Module):
    def __init__(self, d_model):
        super(TimeEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, input_timestamp):
        batch_size, seq_len = input_timestamp.size()

        # Normalize timestamps to a smaller range (e.g., [0, 1])
        input_timestamp = (input_timestamp - input_timestamp.min()) / (input_timestamp.max() - input_timestamp.min())

        # Create time encoding tensor
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=input_timestamp.device)

        # Calculate position and div_term as before
        position = input_timestamp.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=input_timestamp.device).float() *
                             (-math.log(10000.0) / self.d_model))

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe
