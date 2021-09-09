from argparse import Namespace
import copy
import logging
import os
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.classification.frontend.abs_frontend import AbsFrontend


class S3prlFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    def __init__(
        self,
        download_dir: str = None,
        feature_selection: str = "default",
        feat_dim: int = 1024,
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

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wavs = [wav[: input_lengths[i]] for i, wav in enumerate(input)]
        if self.update_extract:
            feats = self.feature_extract(wavs)

        feats_lens = [len(feat) for feat in feats]
        feats = pad_list(feats, 0.0)
        norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        feats = (norm_weights * feats).sum(dim=0)
        feats = torch.transpose(feats, 1, 2) + 1e-6

        feats = self.instance_norm(feats)
        return feats, feats_lens

