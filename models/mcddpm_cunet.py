import torch as th
from models.complex_unet import UNetModel
from utils.mri_data_utils.transform_util import *


class KspaceModel(UNetModel):
    """
    A UNetModel that performs on k-space data. Expects extra kwargs `kspace_zf`, `mask_c`.
    Works directly in k-space domain (no FFT conversions).
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        assert in_channels == 1, "MRI k-space uses 1 complex channel"
        # We use in_channels * 2 because we concatenate x and kspace_zf
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, kspace_zf, mask_c, y=None):
        """
        :param x: [N x 1 x H x W] complex tensor of inputs, x_t at time t (noisy k-space).
        :param timesteps: a batch of timestep indices.
        :param kspace_zf: [N x 1 x H x W] complex tensor, zero-filled k-space measurements.
        :param mask_c: [N x 1 x H x W] complex tensor with value of 0 or 1, equals to 1 - mask.
        :param y: optional [N] tensor of class labels.
        :return: noise estimation or score function in unsampled position of k-space (complex).
        """
        # Concatenate noisy k-space with measurements: [N x 1 x H x W] + [N x 1 x H x W] = [N x 2 x H x W]
        # Both inputs are already in k-space domain
        x_input = th.cat([x, kspace_zf], dim=1)
        output = super().forward(x_input, timesteps, y=y)
        return output * mask_c
