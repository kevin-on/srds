import math
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def parse_sample_type(sample_type_str):
    """Parse sample_type string into method and kwargs dict

    Examples:
        "ddim,eta=0.5" -> {"method": "ddim", "kwargs": {"eta": 0.5}}
        "dir,step=0.1,range=0.5" -> {"method": "dir", "kwargs": {"step": 0.1, "range": 0.5}}
    """
    parts = sample_type_str.split(",")
    method = parts[0].strip()
    kwargs = {}

    if len(parts) > 1:
        for part in parts[1:]:
            part = part.strip()

            # Named parameter: "eta=0.5"
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to convert to appropriate type
            try:
                if "." in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                kwargs[key] = value

    return {"method": method, "kwargs": kwargs}


def decode_latents_to_pil(
    latents: torch.Tensor, pipe: StableDiffusionPipeline
) -> List[Image.Image]:
    images = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    return images


def save_images_as_grid(images: List[Image.Image], output_path: str, grid_cols: int = None):
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
