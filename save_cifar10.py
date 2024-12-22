import os
import hydra

import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from util import get_dataset

SAVE_PATH = '/Home/home/lab05/DiffuseVAE/main/CIFAR_10'
os.makedirs(SAVE_PATH, exist_ok=True)

@hydra.main(config_path="./configs/dataset/cifar10", config_name="train.yaml")
def main(config):
    name = 'cifar10'
    root = config.vae.data.root
    image_size = 32
    dataset = get_dataset(name, root, image_size, norm=False, flip=False)
    
    # This is the CIFAR-10 Dataset
    for idx, img in tqdm(enumerate(dataset)):
        # Img_shape: (C, H, W)
        # print(img.shape)
        if idx > 10000:
            break

        # Convert the tensor to a PIL image
        img = img.numpy().transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))  # Assuming img is in [0, 1]

        print("Saving image to:", SAVE_PATH + f"image_{idx}.png")
        # Save the image to a .png file
        img_pil.save(SAVE_PATH + f"/image_{idx}.png")

if __name__ == '__main__':
    main()
