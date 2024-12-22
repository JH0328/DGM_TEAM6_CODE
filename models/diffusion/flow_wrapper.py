import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from math import log

from models.glow import Glow, calc_z_shapes
    
class VaeFlowWrapper(pl.LightningModule):
    '''
    No Classifer-free guidance,
    No LR scheduling,
    
    '''
    def __init__(
        self,
        vae,
        flow,
        lr=2e-5,
        grad_clip_val=1.0,
        conditional=True,
        eval_mode="sample",
        pred_checkpoints=[],
        n_anneal_steps=0,
        temp=1.0,
        n_bits=5,
        img_size=32
    ):
        super().__init__()
        assert eval_mode in ["sample", "recons"]

        self.vae = vae
        self.flow = flow
        
        # Training arguments
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.conditional = conditional
        self.eval_mode = eval_mode
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp

        # Disable automatic optimization
        self.automatic_optimization = False

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
    
    def calc_flow_loss(self, log_p, logdet, image_size, n_bins):
        # log_p = calc_log_p([z_list])
        n_pixel = image_size * image_size * 3

        loss = -log(n_bins) * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (log(2) * n_pixel)).mean(),
            (log_p / (log(2) * n_pixel)).mean(),
            (logdet / (log(2) * n_pixel)).mean(),
        )
    
    def forward(
        self,
        x,
        cond=None,
        z=None,
        checkpoints=[],
    ):
        print('===================forward=======================')
        # VAE steps
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        x_prime = self.vae.decode(z)
        
        # Flow steps
        # Naive version
        x = self.flow_preprocess(x)
        x_prime = self.flow_preprocess(x_prime)
        
        # if epochs == 0: ??
        log_p, logdet, z_outs = self.flow(x_prime + torch.rand_like(x_prime) / self.n_bins)
        # log_p, logdet, z_outs = self.flow(x + torch.rand_like(x) / self.n_bins)
        # log_p, logdet, z_outs = self.flow(x + torch.rand_like(x) / self.n_bins, x_prime + torch.rand_like(x_prime) / self.n_bins)
        
        return log_p, logdet, z_outs

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()
        
        x = batch
        z = None
        cond = None
        logvar = None
        # VAE Part
        if self.conditional:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x * 0.5 + 0.5)
                var = torch.exp(logvar)
                z = self.vae.reparameterize(mu, logvar)
                cond = self.vae.decode(z)
        
        # Uncertainty Conditioning
        # var shape: [B, 512, 1, 1]
        # cond shape: [B, C, H, w]
        var = var.mean(dim=1, keepdim=True)
        var = F.interpolate(
                var, size=cond.shape[2:], mode='bilinear', align_corners=False
        )
        # Flow Part
        x = self.flow_preprocess(x)
        if self.conditional and cond is not None:
            cond = torch.cat([cond, var], dim=1)
            cond = self.flow_preprocess(cond)
            log_p, logdet, z_outs = self.flow(
                x + torch.rand_like(x) / self.n_bins,
                cond=cond + torch.rand_like(cond) / self.n_bins
            )
        else:
            log_p, logdet, z_outs = self.flow(
                x + torch.rand_like(x) / self.n_bins
            )
        
        # Loss calculation
        logdet = logdet.mean()
        loss, log_p, logdet = self.calc_flow_loss(log_p, logdet, self.img_size, self.n_bins)
        
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.flow.parameters(), self.grad_clip_val
        )
        optim.step()
        lr_sched.step()
        
        self.log("loss", loss, prog_bar=True)
        self.log("log_p", log_p, prog_bar=True)
        self.log("logdet", logdet, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            raise NotImplementedError
        
        # Conditional case
        if self.eval_mode == "sample":
            # batch에서 x_t, z를 받아옴 (여기서 z는 VAE latent로 가정)
            x_t, z = batch
            recons = self.vae(z)
            with torch.no_grad():
                mu, logvar = self.vae.encode(z * 0.5 + 0.5)
            var = torch.exp(logvar)
            var = F.interpolate(
                var, size=recons.shape[2:], mode='bilinear', align_corners=False
            )
            cond = torch.cat([recons, var], dim=1)

            z_shapes = calc_z_shapes(7, self.img_size, self.flow.n_flow, self.flow.n_block)
            z_list = []
            for z_shape in z_shapes:
                z_list.append(torch.randn(recons.size(0), *z_shape, device=recons.device) * self.temp)

            x_recon = self.flow.reverse(z_list, cond=cond)
            return x_recon, recons

        else:
            # Reconstruct
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)

            with torch.no_grad():
                mu, logvar = self.vae.encode(img * 0.5 + 0.5)
            var = torch.exp(logvar)

            var = var.unsqueeze(1)
            var = F.interpolate(
                var, size=recons.shape[2:], mode='bilinear', align_corners=False
            )
            cond = torch.cat([recons, var], dim=1)

            # Flow forward와 reverse 시 cond를 전달
            log_p, logdet, z_outs = self.flow(img, cond=cond)
            x_recon = self.flow.reverse(z_outs, cond=cond)
            return x_recon, recons

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.flow.parameters(), lr=self.lr
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