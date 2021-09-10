from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsClassifier(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
