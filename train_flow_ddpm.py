import copy
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel, UNetModel
from models.vae import VAE
from models.glow import Glow, calc_z_shapes
from util import configure_device, get_dataset

from models.diffusion.flow_wrapper import VaeFlowWrapper
from models.diffusion.vae_flow_ddpm_wrapper import VaeFlowDDPMWrapper

logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="./configs/dataset/cifar10", config_name="train_flow_ddpm.yaml")
def train(config):
    # Get config and setup
    config = config.ddpm
    # config_flow = config.flow
    
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    lr = config.training.lr
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    ddpm_type = config.training.type

    vae = VAE.load_from_checkpoint(
        config.training.vae_chkpt_path,
        input_res=image_size,
    )
    
    # flow = Glow(3, config.model.n_flow, config.model.n_block, affine=config.model.affine, conv_lu=True, cond=True)
    flow = Glow(3, config.model.n_flow, config.model.n_block, affine=config.model.affine, conv_lu=True, cond=True)
    
    # Model 1: VAE_Flow
    vae_flow = VaeFlowWrapper.load_from_checkpoint(
        config.training.vae_flow_chkpt_path,
        vae=vae,
        flow=flow
    )
    
    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        num_heads=config.model.n_heads,
        z_dim=config.training.z_dim,
        use_scale_shift_norm=config.training.z_cond,
        use_z=config.training.z_cond,
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    # vae = VAE.load_from_checkpoint(
    #     config.training.vae_chkpt_path,
    #     input_res=image_size,
    # )
    vae_flow.eval()

    for p in vae_flow.parameters():
        p.requires_grad = False

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")

    ddpm_wrapper = VaeFlowDDPMWrapper(
        online_ddpm,
        target_ddpm,
        vae_flow,
        lr=lr,
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond
    )
    
    # ddpm_wrapper = DDPMWrapper(
    #     online_ddpm,
    #     target_ddpm,
    #     vae,
    #     lr=lr,
    #     cfd_rate=config.training.cfd_rate,
    #     n_anneal_steps=config.training.n_anneal_steps,
    #     loss=config.training.loss,
    #     conditional=False if ddpm_type == "uncond" else True,
    #     grad_clip_val=config.training.grad_clip,
    #     z_cond=config.training.z_cond,
    # )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        # from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
        from pytorch_lightning.strategies import DDPStrategy

        # train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        train_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloaders=loader)

if __name__ == "__main__":
    train()
