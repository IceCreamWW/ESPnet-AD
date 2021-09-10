from argparse import Namespace
import copy
import logging
import os
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.classification.frontend.abs_frontend import AbsFrontend

import pdb


class S3prlFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    def __init__(
        self,
        download_dir: str = None,
        feat_type: str = None,
        feature_selection: str = "hidden_states",
        feat_dim: int = 1024,
        update_extract: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        if download_dir is not None:
            torch.hub.set_dir(download_dir)

        self.feature_selection = feature_selection
        self.feat_dim = feat_dim
        self.extract_feats = torch.hub.load('s3prl/s3prl', feat_type)
        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.update_extract = update_extract

        if not self.update_extract:
            for param in self.extract_feats.parameters():
                param.requires_grad = False

    def get_feat_num(self):
        self.extract_feats.eval()
        wav = [torch.randn(16000).to(next(self.extract_feats.parameters()).device)]
        with torch.no_grad():
            features = self.extract_feats(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def output_size(self) -> int:
        return self.feat_dim

    def get_feats_lens(self, input_lens):
        for conv_layer in self.extract_feats._modules["model"].feature_extractor.conv_layers:
            conv_layer = conv_layer[0]
            input_lens = (input_lens + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) // conv_layer.stride[0] + 1
        return input_lens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wavs = [wav[: input_lengths[i]] for i, wav in enumerate(input)]

        wav_lens = torch.tensor([len(wav) for wav in wavs])

        if not self.update_extract:
            with torch.no_grad():
                feats = self.extract_feats(wavs)[self.feature_selection]
        else:
            feats = self.extract_feats(wavs)[self.feature_selection]


        feats_lens = self.get_feats_lens(wav_lens)
        norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if isinstance(feats, (list, tuple)):
            feats = torch.stack(feats, dim=0)
        else:
            feats = feats.unsqueeze(0)
        pdb.set_trace()
        feats = (norm_weights * feats).sum(dim=0)
        feats = torch.transpose(feats, 1, 2) + 1e-6
        feats = self.instance_norm(feats)
        return feats, feats_lens

