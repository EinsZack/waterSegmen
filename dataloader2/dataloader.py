import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import


class WaterSceneDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load image
        image_filename = self.image_filenames[index]
        image = Image.open(image_filename).convert('RGB')

        # Load annotation
        annotation_filename = image_filename.replace('.jpg', '.png')
        annotation = Image.open(annotation_filename).convert('L')
        image_anno = np.array(annotation)

        # Apply transformations
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        annotation = torch.tensor(np.array(annotation), dtype=torch.long)
        # annotation = torch.tensor(np.array(annotation)).unsqueeze(0)

        return image, annotation


def create_data_loader(data_dir, batch_size, shuffle=True, num_workers=4):
    dataset = WaterSceneDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
