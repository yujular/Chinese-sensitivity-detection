from torch import nn
from torch.nn import functional as F


class CNNAdapter(nn.Module):
    def __init__(self, max_length, hidden_size, dropout_rate=0.1):
        """
        CNN Adapter, input shape: (N, 1, max_length, hidden_size)
        :param max_length: max length of token sequence
        :param hidden_size: hidden size of bert model
        :param dropout_rate: dropout rate
        """
        super(CNNAdapter, self).__init__()

        self.input_length = max_length
        self.input_width = hidden_size
        self.dropout_rate = dropout_rate

        self.layer_norm = nn.LayerNorm([self.input_length, self.input_width])

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, padding=1)
        self.avg_pool1 = nn.AvgPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.avg_pool2 = nn.AvgPool2d((2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.avg_pool3 = nn.AvgPool2d((2, 2))

        self.extra = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1), stride=8, padding=0),
            nn.BatchNorm2d(64)
        )

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.bn = nn.BatchNorm1d(64)
        self.num_features = self.feature_num()

    def forward(self, x):
        # (N,l,d)
        x = self.layer_norm(x)
        # (N,1,l,d), for conv2d
        x = x.unsqueeze(1)

        # residual
        residual = x
        residual = self.extra(residual)

        x = F.relu(self.avg_pool1(self.conv1(x)))
        x = F.relu(self.avg_pool2(self.conv2(x)))
        x = F.relu(self.avg_pool3(self.conv3(x)))

        x = F.relu(x + residual)

        # global pooling
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        x = x_max + x_avg
        # Flatten
        x = x.view(x.size(0), -1)
        # dropout and bn
        x = self.dropout(x)
        x = self.bn(x)
        return x

    def feature_num(self):
        return self.bn.num_features
