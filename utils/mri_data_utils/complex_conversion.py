import torch as th


def to_complex(x):
    """
    Convert [N x 2 x H x W] real to [N x 1 x H x W] complex using PyTorch's view_as_complex.
    If already complex, return as-is.
    
    :param x: Input tensor, either [N x 2 x H x W] real or [N x 1 x H x W] complex
    :return: [N x 1 x H x W] complex tensor
    """
    # Already complex, nothing to do
    if x.is_complex():
        return x
    
    # Check if it's 2-channel real (real/imag representation)
    if x.shape[1] == 2:
        # Reshape to [N, H, W, 2] for view_as_complex
        x = x.permute(0, 2, 3, 1)  # [N, 2, H, W] -> [N, H, W, 2]
        x = th.view_as_complex(x.contiguous())  # [N, H, W] complex
        x = x.unsqueeze(1)  # [N, 1, H, W] complex
        return x
    
    # Single channel or unexpected shape, return as-is
    return x
