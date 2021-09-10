import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.classification.classifier.abs_classifier import AbsClassifier
from espnet2.classification.layers.pooling import AttentiveStatisticsPooling, StatisticsPooling

class TDNN(AbsClassifier):
    """
    The extended TDNN xvec described in
    SPEAKER RECOGNITION FOR MULTI-SPEAKER CONVERSATIONS USING X-VECTORS
    and
    State-of-the-art speaker recognition with neural network embeddings in NIST SRE18 and Speakers in the Wild evaluations
    Which is just a deeper xvec architecture
    """
    def __init__(self, feat_dim=40, hid_dim=512, stats_dim=1500, emb_dim=512, num_classes=2,
                 pooling='statistic', stddev=True, affine_layers=2, norm_way='softmax'):
        super(TDNN, self).__init__()
        self.frame_1 = TDNN_Layer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_1a = TDNN_Layer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_2 = TDNN_Layer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_2a = TDNN_Layer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_3 = TDNN_Layer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_3a = TDNN_Layer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_4 = TDNN_Layer(hid_dim, hid_dim, context_size=3, dilation=4)
        self.frame_4a = TDNN_Layer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = TDNN_Layer(hid_dim, stats_dim, context_size=1, dilation=1)

        if pooling == 'statistic':
            self.pooling = StatisticsPooling(stddev=stddev)
        else:
            self.pooling = AttentiveStatisticsPooling(stats_dim, affine_layers, stddev, norm_way)

        self.seg_1 = nn.Linear(stats_dim * 2, emb_dim)
        self.seg_bn_1 = nn.BatchNorm1d(emb_dim, affine=False)
        self.seg_2 = nn.Linear(emb_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, x, x_lens):
        out, out_lens = self.frame_1(x, x_lens)
        out, out_lens = self.frame_1a(out, out_lens)
        out, out_lens = self.frame_2(out, out_lens)
        out, out_lens = self.frame_2a(out, out_lens)
        out, out_lens = self.frame_3(out, out_lens)
        out, out_lens = self.frame_3a(out, out_lens)
        out, out_lens = self.frame_4(out, out_lens)
        out, out_lens = self.frame_4a(out, out_lens)
        out, out_lens = self.frame_5(out, out_lens)

        # pooling_mean = torch.mean(out, dim=2)
        # pooling_std = torch.sqrt(torch.var(out, dim=2) + 1e-8)
        # stats = torch.cat((pooling_mean, pooling_std), 1)
        stats = self.pooling(out, out_lens)
        embed_a = self.seg_1(stats)
        out = F.relu(embed_a)
        out = self.seg_bn_1(out)
        embed_b = self.seg_2(out)
        out = self.linear(embed_b)
        return out, out_lens


class TDNN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """
        Define a single time delay layer, it's actually a 1-D convolution
        Here, we treat the feat_dim as the initial channal numbers
        in_dim and out_dim corresponds to the number of input and output channels, respectively
        context_size and dilation defines the real context used for computation:
        eg: [-2,2] is represented as context_size=5, dilation=1
            {-2,0,2} is represented as context_size=3, dilation=2
        """
        super(TDNN_Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(self.in_dim, self.out_dim, self.context_size, dilation=self.dilation, padding=self.padding)
        # Affine=false is to be compatible with the original kaldi implementation
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x, x_lens):
        out_lens = x_lens + 2 * self.padding - self.dilation * (self.context_size - 1)
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out, out_lens



if __name__ == '__main__':
    # net = XVEC(feat_dim=40, hid_dim=512, stats_dim=1500, emb_dim=512)
    net = EXT_XVEC(feat_dim=40, hid_dim=512, stats_dim=1500, emb_dim=256)
    x = torch.randn(16, 40, 500)
    embed_a, embed_b = net(x)
    print(embed_a.shape)
    print(embed_b.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
