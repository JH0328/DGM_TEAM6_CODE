import logging
import os

import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

# from models.vae import VAE
from models.vae import VAE
from models.glow import Glow, calc_z_shapes
from util import configure_device, get_dataset

from models.diffusion.flow_wrapper import VaeFlowWrapper

logger = logging.getLogger(__name__)

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

@hydra.main(config_path="./configs/dataset/cifar10", config_name="train_flow.yaml")
def train(config):
    # Get config and setup
    config = config.flow
    logger.info(OmegaConf.to_yaml(config))

        # self,
        # vae,
        # flow,
        # lr=2e-5,
        # grad_clip_val=1.0,
        # conditional=True,
        # eval_mode="sample",
        # pred_checkpoints=[],
        # temp=1.0,
        # n_bits=5,
        # img_size=32
        
    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Arguments
    lr = config.training.lr
    
    # Model 1: Flow
    flow = Glow(3, config.model.n_flow, config.model.n_block, affine=config.model.affine, conv_lu=True, cond=True)
    
    vae = VAE.load_from_checkpoint(
        config.training.vae_chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False

    vaeflow_wrapper = VaeFlowWrapper(
        vae,
        flow,
        lr=lr,
        img_size=image_size,
        n_anneal_steps=config.training.n_anneal_steps
    )
    
    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vaeflow-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        from pytorch_lightning.strategies import DDPStrategy

        train_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

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
    trainer.fit(vaeflow_wrapper, train_dataloaders=loader)

if __name__ == "__main__":
    train()
