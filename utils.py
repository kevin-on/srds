import math
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def decode_latents_to_pil(
    latents: torch.Tensor, pipe: StableDiffusionPipeline
) -> List[Image.Image]:
    images = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    return images


def save_images_as_grid(
    images: List[Image.Image], output_path: str, grid_cols: int = None
):
    """Save multiple PIL images as a grid."""
    if not images:
        return

    if grid_cols is None:
        grid_cols = math.ceil(math.sqrt(len(images)))
    grid_rows = math.ceil(len(images) / grid_cols)

    img_width, img_height = images[0].size
    grid_width = grid_cols * img_width
    grid_height = grid_rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))

    grid_image.save(output_path)
    return grid_image
