# Helper script to sample from a conditional DDPM model
# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import copy

import hydra
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from datasets.latent import LatentDataset
from models.callbacks import ImageWriter
from util import configure_device

from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from models.glow import Glow, calc_z_shapes

from models.diffusion.flow_wrapper import VaeFlowWrapper
from models.diffusion.vae_flow_ddpm_wrapper import VaeFlowDDPMWrapper

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

os.makedirs('./results_ddpm', exist_ok=True)
# os.makedirs('./results_vae', exist_ok=True)

@hydra.main(config_path="./configs/dataset/cifar10", config_name="test_flow_ddpm.yaml")
# @hydra.main(config_path=os.path.join(p, "configs"))
def sample_cond(config):
    # Seed and setup
    config_ddpm = config.ddpm
    config_vae = config.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    
    flow = Glow(3, config_ddpm.model.n_flow, config_ddpm.model.n_block, affine=config_ddpm.model.affine, conv_lu=True, cond=True)

    # Model 1: VAE_Flow
    vae_flow = VaeFlowWrapper.load_from_checkpoint(
        config_ddpm.evaluation.vae_flow_chkpt_path,
        vae=vae,
        flow=flow
    )
    vae_flow.eval()
    
    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=3,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
        z_dim=config_ddpm.evaluation.z_dim,
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    # ddpm_wrapper = VaeFlowDDPMWrapper(
    #     online_ddpm,
    #     target_ddpm,
    #     vae_flow,
    #     lr=lr,
    #     cfd_rate=config.training.cfd_rate,
    #     n_anneal_steps=config.training.n_anneal_steps,
    #     loss=config.training.loss,
    #     conditional=False if ddpm_type == "uncond" else True,
    #     grad_clip_val=config.training.grad_clip,
    #     z_cond=config.training.z_cond
    # )
    
    ddpm_wrapper = VaeFlowDDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae_flow=vae_flow,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="sample",
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        strict=True,
        ddpm_latents=ddpm_latents,
    )

    # Create predict dataset of latents
    z_dataset = LatentDataset(
        (n_samples, config_vae.model.z_dim, 1, 1),
        (n_samples, 3, image_size, image_size),
        share_ddpm_latent=True if ddpm_latent_path != "" else False,
        expde_model_path=config_vae.evaluation.expde_model_path,
        seed=config_ddpm.evaluation.seed,
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        config_ddpm.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_vae=config_ddpm.evaluation.save_vae,
        save_mode=config_ddpm.evaluation.save_mode,
        is_norm=config_ddpm.data.norm,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)

if __name__ == "__main__":
    sample_cond()
