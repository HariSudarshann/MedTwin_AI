"""
MediTwin AI — TorchXRayVision Chest X-Ray Model

Wrapper around the pretrained TorchXRayVision DenseNet-121 model.
"""

import logging
from typing import Dict, Any

import numpy as np
import torch
import torchvision
import skimage.io
import torchxrayvision as xrv

logger = logging.getLogger(__name__)


class TorchXRayVisionClassifier:
    """Chest X-ray inference using TorchXRayVision DenseNet-121."""

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        device: str | None = None,
    ):
        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        logger.info(
            "Loading TorchXRayVision model '%s' on %s",
            weights,
            self.device,
        )

        self.model = xrv.models.DenseNet(weights=weights)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.pathologies = list(self.model.pathologies)

        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

    def preprocess(self, image) -> torch.Tensor:
        """Convert a PIL/numpy image to TorchXRayVision input format."""

        if hasattr(image, "convert"):
            # PIL image
            image = np.asarray(image.convert("L"))

        image = np.asarray(image)

        # Ensure grayscale
        if image.ndim == 3:
            image = image.mean(axis=2)

        # Normalize according to TorchXRayVision convention
        image = xrv.datasets.normalize(image, 255)

        # Add channel dimension
        image = image[None, ...]

        # Resize/crop
        image = self.transform(image)

        tensor = torch.from_numpy(image).float().unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, image) -> Dict[str, Any]:
        """Run inference and return pathology scores."""

        tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(tensor)

        scores = output[0].detach().cpu().numpy()

        findings = {
            pathology: round(float(score), 4)
            for pathology, score in zip(
                self.pathologies,
                scores,
            )
        }

        return {
            "model": "TorchXRayVision DenseNet-121",
            "weights": "densenet121-res224-all",
            "device": str(self.device),
            "findings": findings,
            "pathologies": self.pathologies,
        }