import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import skimage.io
import torchxrayvision as xrv


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/test_torchxrayvision.py path/to/xray.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading image: {image_path}")

    # Load grayscale / image
    img = skimage.io.imread(str(image_path))

    # Normalize to XRV expected range
    img = xrv.datasets.normalize(img, 255)

    # Convert to single channel [1, H, W]
    if img.ndim == 3:
        img = img.mean(2)

    img = img[None, ...]

    # Official-style preprocessing
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    img = transform(img)
    img = torch.from_numpy(img).float()

    # Load pretrained model
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    results = dict(
        zip(
            model.pathologies,
            output[0].detach().cpu().numpy()
        )
    )

    print("\nChest X-ray findings:")
    print("-" * 45)

    for pathology, score in results.items():
        print(f"{pathology:30s}: {score:.4f}")

    print("\nDevice:", device)


if __name__ == "__main__":
    main()