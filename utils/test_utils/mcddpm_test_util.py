from utils.test_utils.ddpm_test_util import DDPMTestLoop
from utils import dist_util
from utils.mri_data_utils.transform_util import *
from utils.mri_data_utils.complex_conversion import to_complex


class MCDDPMTestLoop(DDPMTestLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, batch_kwargs):
        # Convert 2-channel real to complex
        kspace_zf = to_complex(batch_kwargs["kspace_zf"].to(dist_util.dev()))
        mask_c = to_complex(batch_kwargs["mask_c"].to(dist_util.dev()))
        
        cond = {
            "kspace_zf": kspace_zf,
            "mask_c": mask_c,
        }
        samples = []
        while len(samples) * self.batch_size < self.num_samples_per_mask:
            sample = self.diffusion.sample_loop(
                self.model,
                (self.batch_size, 1, self.image_size, self.image_size),  # 1 complex channel
                cond,
                clip=False
            )
            # sample is already k-space (no FFT needed with new KspaceModel)
            kspace = sample + cond["kspace_zf"]
            # Convert to image space for saving
            sample = ifftc_th(kspace)
            samples.append(sample.cpu().numpy())

        # gather all samples and save them
        samples = np.concatenate(samples, axis=0)
        samples = samples[: self.num_samples_per_mask]
        return samples
