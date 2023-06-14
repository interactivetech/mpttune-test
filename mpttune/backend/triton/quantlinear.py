import torch

from mpttune.backend.base import QuantLinearBase
import mpttune.backend.triton.triton_utils as tu
from mpttune.backend.triton.autograd import AutogradMatmul


class QuantLinear(QuantLinearBase):
    framework = 'triton'

    def forward(self, x):
        if torch.is_grad_enabled():
            out = AutogradMatmul.apply(
                x, self.qweight, self.scales,
                self.qzeros, self.g_idx, self.bits, self.maxq)
        else:
            out = self._forward_no_grad(x)

        if self.bias is not None:
            out += self.bias

        return out

    def _forward_no_grad(self, x):
        assert self.qzeros.dtype == torch.int32
        return tu.triton_matmul(x, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
