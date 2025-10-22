"""
PickScore: A learned scoring function for text-image alignment
Reference: https://github.com/yuvalkirstain/PickScore
"""

from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class PickScoreInferencer:
    """
    PickScore Inferencer for scoring text-image alignment.

    This class provides an interface to calculate PickScore scores for images
    given a text prompt. It can be used in spararealtts.py for sample selection.
    """

    def __init__(self, device="cuda"):
        """
        Initialize PickScore model.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device

        # Model paths
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    @torch.no_grad()
    def score(self, images: List[Image.Image], prompt: str) -> List[float]:
        """
        Calculate PickScore scores for a list of images given a prompt.

        Args:
            images: List of PIL Image objects
            prompt: Text prompt to compare against

        Returns:
            scores: List of float scores (higher is better)
        """
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Embed images
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        # Embed text
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Calculate scores
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        return scores.cpu().tolist()

    @torch.no_grad()
    def calc_probs(self, images: List[Image.Image], prompt: str) -> List[float]:
        """
        Calculate softmax probabilities for a list of images given a prompt.
        Useful when you want to compare multiple images and get normalized probabilities.

        Args:
            images: List of PIL Image objects
            prompt: Text prompt to compare against

        Returns:
            probs: List of probability values (sum to 1.0)
        """
        # Get raw scores
        scores = self.score(images, prompt)
        scores_tensor = torch.tensor(scores)

        # Apply softmax
        probs = torch.softmax(scores_tensor, dim=-1)

        return probs.cpu().tolist()


if __name__ == "__main__":
    # Example usage
    inferencer = PickScoreInferencer(device="cuda")

    # Load example images
    pil_images = [
        Image.open(
            "/home2/junoh/2025_para_scaling/srds/output/sparatts_test/20251013_080911_cs10-fs100/srds_final.png"
        ),
        # Image.open("/home2/junoh/2025_para_scaling/srds/output/sparatts_test/20251013_074203_cs10-fs100/srds_final.png"),
        # Image.open("/home2/junoh/2025_para_scaling/srds/output/sparatts_test/20251013_074003_cs10-fs100/srds_iteration_9.png")
    ]
    prompt = "a photo of a dog and a cat"

    # Calculate scores
    scores = inferencer.score(pil_images, prompt)
    print(f"PickScores: {scores}")

    # Calculate probabilities
    probs = inferencer.calc_probs(pil_images, prompt)
    print(f"Probabilities: {probs}")
