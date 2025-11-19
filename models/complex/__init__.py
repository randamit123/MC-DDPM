from torch.nn import (
    Conv1d as ComplexConv1d,
    Conv2d as ComplexConv2d,
)

from torchcvnn.nn import (
    ConvTranspose2d as ComplexConvTranspose2d,
    BatchNorm1d as ComplexBatchNorm1d,
    BatchNorm2d as ComplexBatchNorm2d,
    Dropout as ComplexDropout,
    Dropout2d as ComplexDropout2d,
    MaxPool2d as ComplexMaxPool2d,
    AvgPool2d as ComplexAvgPool2d,
    Upsample as ComplexUpsample,
    CReLU as ComplexReLU,
    CPReLU as ComplexPReLU,
    CSigmoid as ComplexSigmoid,
    CTanh as ComplexTanh,
)


def complex_conv_nd(dims, in_channels, out_channels, kernel_size, dtype=None, device=None, **kwargs):
    """
    Create a 1D, 2D, or 3D complex convolution module.
    """
    import torch
    if dtype is None:
        dtype = torch.complex64
    
    if dims == 1:
        conv = ComplexConv1d(in_channels, out_channels, kernel_size, 
                            dtype=dtype, device=device, **kwargs)
    elif dims == 2:
        conv = ComplexConv2d(in_channels, out_channels, kernel_size,
                            dtype=dtype, device=device, **kwargs)
    elif dims == 3:
        raise NotImplementedError("ComplexConv3d not implemented in this library")
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
    
    return conv


__all__ = [
    'ComplexConv1d',
    'ComplexConv2d',
    'ComplexConvTranspose2d',
    'ComplexBatchNorm1d',
    'ComplexBatchNorm2d',
    'ComplexReLU',
    'ComplexSigmoid',
    'ComplexTanh',
    'ComplexPReLU',
    'ComplexMaxPool2d',
    'ComplexAvgPool2d',
    'ComplexDropout',
    'ComplexDropout2d',
    'ComplexUpsample',
    'complex_conv_nd',
]
