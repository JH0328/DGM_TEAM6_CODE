import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from math import log

from models.glow import Glow, calc_z_shapes

from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from util import space_timesteps

class VaeFlowDDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae_flow,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
        img_size=32,
        n_bits=5,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.vae = vae_flow.vae
        # self.vae = vae
        self.flow = vae_flow.flow
        
        self.online_network = online_network
        self.target_network = target_network
        
        self.z_cond = z_cond
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None
        
        # Flow arguments
        self.n_bits = n_bits
        self.n_bins = 2.0 ** n_bits
        self.img_size = img_size

    def flow_preprocess(self, x):
        x = x * 255
        if self.n_bits < 8:
            x = torch.floor(x/2 ** (8 - self.n_bits))
            
        x = x / self.n_bins - 0.5
        
        return x
    
    def flow_deprocess(self, x):
        # flow_preprocess에서 마지막 단계: x = x / self.n_bins - 0.5
        # 역연산: x = (x + 0.5) * self.n_bins
        x = (x + 0.5) * self.n_bins  # (B, C, N), 값 범위: [0, n_bins)

        # flow_preprocess에서 n_bits < 8인 경우
        #   x = floor((original * 255) / 2^(8 - n_bits))
        # 역으로 복원하려면 floor 부분은 손실이 있지만 스케일만 역으로 맞추어 준다.
        if self.n_bits < 8:
            # original * 255 = x * 2^(8 - self.n_bits)
            x = x * (2 ** (8 - self.n_bits))
        # 이 시점에서 x는 (original * 255)에 해당하는 값

        # flow_preprocess에서 처음: original -> original * 255
        # 역연산: original = x / 255
        x = x / 255.0

        return x

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        logvar = None
        x = batch
        
        # if self.conditional:
        #     x = batch
        #     with torch.no_grad():
        #         mu, logvar = self.vae.encode(x * 0.5 + 0.5)
        #         var = torch.exp(logvar)
        #         z = self.vae.reparameterize(mu, logvar)
        #         cond = self.vae.decode(z)
        #         cond = 2 * cond - 1
                # print(cond.min())
                # print(cond.max())
                # print('===')

            # Set the conditioning signal based on clf-free guidance rate
            # if torch.rand(1)[0] < self.cfd_rate:
            #     cond = torch.zeros_like(x)
            #     z = torch.zeros_like(z)
        # else:
            # x = batch

        # Flow part
        # var = var.mean(dim=1, keepdim=True)
        # var = F.interpolate(
        #         var, size=cond.shape[2:], mode='bilinear', align_corners=False
        # )
        # cond = torch.cat([cond, var], dim=1)
        
        # x = self.flow_preprocess(x)
        
        z_shapes = calc_z_shapes(7, self.img_size, self.flow.n_flow, self.flow.n_block)
        z_list = []
        for z_shape in z_shapes:
            z_list.append(torch.randn(x.size(0), *z_shape, device=x.device) * self.temp)

        with torch.no_grad():
            x_recon = self.flow.reverse(z_list).data
            if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
                # print("NaN/Inf detected in flow reverse.")
                # Skip this batch
                return torch.tensor(0.0, requires_grad=True, device=x.device)
        
        B = x_recon.shape[0]
        x_flat = x_recon.view(B, -1)

        x_min = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x_max = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        x_recon = 2 * (x_recon - x_min) / (x_max - x_min) - 1
        
        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=x_recon, z=None
        )
        
        # eps_pred = self.online_network(
            # x, eps, t, low_res=cond, z=None
        # )
        
        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            
            z_shapes = calc_z_shapes(7, self.img_size, self.flow.n_flow, self.flow.n_block)
            z_list = []
            for z_shape in z_shapes:
                z_list.append(torch.randn(x_t.size(0), *z_shape, device=x_t.device) * self.temp)
            
            with torch.no_grad():
                x_recon = self.flow.reverse(z_list).data
                if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
                    self.predict_step(batch, batch_idx, dataloader_idx) # do recursively
            
            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            # if isinstance(self.online_network, DDPMv2):
            #     x_t = x_recon + self.temp * torch.randn_like(x_recon)
        else:
            raise NotImplementedError
        
            # img = batch
            # recons = self.vae.forward_recons(img * 0.5 + 0.5)
            # recons = 2 * recons - 1

            # # DDPM encoder
            # x_t = self.online_network.compute_noisy_input(
            #     img,
            #     torch.randn_like(img),
            #     torch.tensor(
            #         [self.online_network.T - 1] * img.size(0), device=img.device
            #     ),
            # )

            # if isinstance(self.online_network, DDPMv2):
            #     x_t += recons

        return (
            self(
                x_t,
                cond=x_recon,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            x_recon,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }