"""EvSnowNet paper variant with event-predicted adaptive image fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evsnownet_backbone import Transformer as _EvSnowNetBackbone


class Transformer(_EvSnowNetBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_head = nn.Conv2d(self.dim[0], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_img = x[:, :3, :, :]
        x_events = x[:, 3:, :, :]

        # Keep event preprocessing aligned with shared EvSnowNet backbone behavior.
        x_events = self.check_image_size(x_events)
        evt_pyr = self.eventnet(x_events)

        # Reuse backbone restoration as the reconstruction candidate.
        recon = self.snowformer_model_forward(x_img, evt_pyr)

        mask_logits = self.mask_head(evt_pyr["s1"])
        mask = torch.sigmoid(mask_logits)
        if mask.shape[2:] != recon.shape[2:]:
            mask = F.interpolate(mask, size=recon.shape[2:], mode="bilinear", align_corners=False)
            # mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)  # Normalize to [0,1]
            # # self.mask_smooth_kernel = 5   # odd number
            # # self.mask_smooth_passes = 1
            
            # # # Smooth mask after interpolation.
            # # for _ in range(self.mask_smooth_passes):
            # #     mask = F.avg_pool2d(
            # #         mask,
            # #         kernel_size=self.mask_smooth_kernel,
            # #         stride=1,
            # #         padding=self.mask_smooth_kernel // 2,
            # #     )
    

        # Adaptive fusion: input*(1-mask) + reconstruction*mask.
        out = torch.lerp(x_img, recon, mask)
        return out, evt_pyr, mask, recon
