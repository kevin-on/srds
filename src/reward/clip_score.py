from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPScoreInferencer:
    """
    CLIP Score Inferencer for measuring text-image alignment.
    Uses CLIP model to compute similarity between text and images.
    """

    def __init__(self, device="cuda", model_name="openai/clip-vit-base-patch32"):
        """
        Initialize CLIP Score Inferencer.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_name: CLIP model name to use
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        print(f"CLIP model loaded successfully on {device}")

    @torch.no_grad()
    def score(self, images: List[Image.Image], prompt: str) -> List[float]:
        """
        Compute CLIP scores between images and text prompt.

        Args:
            images: List of PIL Images
            prompt: Text prompt string

        Returns:
            List of CLIP scores (higher is better)
        """
        # Process inputs
        inputs = self.processor(
            text=[prompt] * len(images), images=images, return_tensors="pt", padding=True
        ).to(self.device)

        # Get features
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # [batch_size, 1]

        # Return raw logits (cosine similarity * temperature)
        scores = logits_per_image.squeeze().cpu()
        if scores.dim() == 0:  # Single image case
            return [scores.item()]
        else:  # Multiple images case
            return scores.tolist()

    @torch.no_grad()
    def raw_similarity(self, images: List[Image.Image], prompt: str) -> List[float]:
        """
        Get raw cosine similarity scores (without softmax).

        Args:
            images: List of PIL Images
            prompt: Text prompt string

        Returns:
            List of raw cosine similarity scores
        """
        # Process inputs
        inputs = self.processor(
            text=[prompt] * len(images), images=images, return_tensors="pt", padding=True
        ).to(self.device)

        # Get features
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # [batch_size, 1]

        return logits_per_image.cpu().squeeze().tolist()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from PIL import Image

    # Create some dummy images for testing
    dummy_images = []
    for i in range(3):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        dummy_images.append(img)

    prompt = "a beautiful landscape"

    # Initialize inferencer
    clip_inferencer = CLIPScoreInferencer(device="cuda")

    # Compute scores
    scores = clip_inferencer.score(dummy_images, prompt)
    raw_sim = clip_inferencer.raw_similarity(dummy_images, prompt)

    print(f"Prompt: {prompt}")
    print(f"CLIP scores: {scores}")
    print(f"Raw similarities: {raw_sim}")
