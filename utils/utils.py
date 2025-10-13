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


def save_images_as_grid(
    images: List[Image.Image], 
    output_path: str, 
    grid_cols: int = None,
    selected_idx: int = None,
    reward_idx: int = None
):
    """Save multiple PIL images as a grid.
    
    Args:
        images: List of PIL images to arrange in a grid
        output_path: Path to save the grid image
        grid_cols: Number of columns in the grid (default: sqrt of number of images)
        selected_idx: Index of the selected image to highlight with red border (optional)
        reward_idx: Index of the best reward score image to highlight with blue border (optional)
    """
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
        
        img_copy = img.copy()
        needs_border = False
        
        # Add borders if this image is selected or has best reward score
        if selected_idx is not None and idx == selected_idx:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img_copy)
            border_width = 10
            # Draw red border (selected by distance metric)
            for i in range(border_width):
                draw.rectangle(
                    [i, i, img_width - 1 - i, img_height - 1 - i],
                    outline="red"
                )
            needs_border = True
        
        if reward_idx is not None and idx == reward_idx:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img_copy)
            border_width = 10
            # If it's also the selected one, draw blue border inside red border
            if selected_idx is not None and idx == selected_idx:
                # Draw blue border inside the red border
                for i in range(border_width, border_width * 2):
                    draw.rectangle(
                        [i, i, img_width - 1 - i, img_height - 1 - i],
                        outline="blue"
                    )
            else:
                # Draw blue border only
                for i in range(border_width):
                    draw.rectangle(
                        [i, i, img_width - 1 - i, img_height - 1 - i],
                        outline="blue"
                    )
            needs_border = True
        
        if needs_border:
            grid_image.paste(img_copy, (x, y))
        else:
            grid_image.paste(img, (x, y))

    grid_image.save(output_path)
    return grid_image
