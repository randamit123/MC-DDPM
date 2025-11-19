from utils.train_utils.ddpm_train_util import *
from utils.mri_data_utils.complex_conversion import to_complex


class MCDDPMTrainLoop(DDPMTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        # modify condition so that it only contains the information we need.
        batch, cond = batch
        
        # Convert 2-channel real to complex tensors if needed
        batch = to_complex(batch)
        cond["kspace_zf"] = to_complex(cond["kspace_zf"])
        cond["mask_c"] = to_complex(cond["mask_c"])
        
        # Remove image_zf - no longer needed
        cond = {
            k: cond[k] for k in ["kspace_zf", "mask_c"]
        }
        return batch, cond
