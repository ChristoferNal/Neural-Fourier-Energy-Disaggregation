import torch
from torch import nn
from torchnlp.nn.attention import Attention


class FourierBLock(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.0, mode='fft', leaky_relu=False):
        """
        Input arguments:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
            mode - The type of mechanism inside the block. Currently, three types are supported; 'fft' for fourier,
            'att' for dot attention and 'plain' for simple concatenation.
                default value: 'fft'
            leaky_relu - A flag that controls whether leaky relu should be applied on the linear layer after the
            fourier mechanism.
                default value: False
        """
        super().__init__()
        self.mode = mode
        if self.mode == 'att':
            self.attention = Attention(input_dim, attention_type='dot')

        if leaky_relu:
            self.linear_fftout = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.linear_fftout = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
            )

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        fft_out = self.norm1(x)
        if self.mode == 'fft':
            fft_out = torch.fft.fft(fft_out, dim=-1)
            fft_out = torch.cat((fft_out.real, fft_out.imag), dim=-1)
        elif self.mode == 'att':
            fft_out, _ = self.attention(fft_out, fft_out)
            fft_out = torch.cat((fft_out, fft_out), dim=-1)
        elif self.mode == 'plain':
            fft_out = torch.cat((fft_out, fft_out), dim=-1)

        fft_out = self.linear_fftout(fft_out)
        x = x + self.dropout(fft_out)
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        return x


class LinearDropRelu(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(LinearDropRelu, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear(x)


class ConvDropRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, groups=1):
        super(ConvDropRelu, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class NFED(nn.Module):
    def __init__(self, depth, kernel_size, cnn_dim, **block_args):
        """
        Input arguments:
            depth - The number of fourier blocks in series
            kernel_size - The kernel size of the first CNN layer
            cnn_dim - Dimensionality of the output of the first CNN layer
        """
        super(NFED, self).__init__()
        self.drop = block_args['dropout']
        self.input_dim = block_args['input_dim']
        self.dense_in = self.input_dim * cnn_dim // 2

        self.conv = ConvDropRelu(1, cnn_dim, kernel_size=kernel_size, dropout=self.drop)
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.fourier_layers = nn.ModuleList([FourierBLock(**block_args) for _ in range(depth)])

        self.flat = nn.Flatten()
        self.dense1 = LinearDropRelu(self.dense_in, cnn_dim, self.drop)
        self.dense2 = LinearDropRelu(cnn_dim, cnn_dim // 2, self.drop)

        self.output = nn.Linear(cnn_dim // 2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out
