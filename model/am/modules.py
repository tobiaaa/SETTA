import torch
import torch.nn as nn
import torch.nn.functional as F


class InputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(201, 256)
        self.lin_2 = nn.Linear(256, 256)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        x = x.transpose(-1, -2)
        
        return x

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_1 = nn.LayerNorm(256)
        self.norm_2 = nn.LayerNorm(256)
        self.time_block = TimeBlock()
        self.freq_block_1 = FrequencyBlock()
        self.freq_block_2 = FrequencyBlock()
        self.local_block = LocalBlock()

    def forward(self, x):
        x_in = x.clone()
        x = x.transpose(-1, -2)
        x = self.norm_1(x)
        x_t = self.time_block(x)
        x = x + x_t

        x_f = self.freq_block_1(x)
        x = x + x_f

        x_l = self.local_block(x)
        x = x + x_l

        x = self.norm_2(x)
        x = self.freq_block_2(x)
        x = x.transpose(-1, -2)

        x = x + x_in

        return x


class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(256, 256)
        self.lin_2 = nn.Linear(256, 201)

    def forward(self, x):
        x = x.transpose(-1, -2)

        x = self.lin_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        
        x = x.transpose(-1, -2)
        return x


class LocalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 16, 5, padding='same')
        self.multi_dil = MultiDilation(16)
        self.conv_2 = nn.Conv2d(16, 1, 5, padding='same')

    def forward(self, x):
        x = x[:, None]
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.multi_dil(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = x.squeeze(1)
        return x

class MultiDilation(nn.Module):
    def __init__(self, c_in, n_dil=4):
        super().__init__()
        layers = []
        for i in range(n_dil):
            dil = 2 ** i
            layers.append(
                nn.Conv2d(c_in, c_in, 3, dilation=dil, padding='same')
            )

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Conv2d(c_in * n_dil, c_in, 1)

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(layer(x))

        x = torch.concat(out, 1)
        x = F.relu(x)
        x = self.output_layer(x)
        return x


class TimeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_q = nn.Linear(256, 256)
        self.lin_k = nn.Linear(256, 256)

    def forward(self, x):
        q = self.lin_q(x)
        k = self.lin_k(x)
        x = F.scaled_dot_product_attention(q, k, x)
        return x


class FrequencyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(256, 256)
        self.lin_2 = nn.Linear(256, 256)


    def forward(self, x):
        x_gate = self.lin_1(x)
        x = x * x_gate
        x = self.lin_2(x)
        
        return x
