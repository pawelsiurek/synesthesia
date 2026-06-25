import open_clip
import torch
from PIL import Image
import numpy as np
from pathlib import Path


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"CLIP loaded on {self.device}")

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)  # L2 normalize

        return features.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)  # L2 normalize

        return features.cpu().numpy().flatten().astype(np.float32)

    def encode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().astype(np.float32)