from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class LineDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, pair["text"]
