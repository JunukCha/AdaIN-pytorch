import os, os.path as osp
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader

Image.MAX_IMAGE_PIXELS = None

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, num_data=None, test=False):
        self.root = root
        self.transform = transform
        self.num_data = num_data
        self.test = test

        self.content_paths = sorted([
            osp.join(root, "content", fname) for fname in os.listdir(osp.join(root, "content"))
        ])
        self.style_paths = sorted([
            osp.join(root, "style", fname) for fname in os.listdir(osp.join(root, "style"))
        ])

    def __len__(self):
        if self.num_data is not None:
            return self.num_data
        else:
            return len(self.content_paths)
    
    def __getitem__(self, index):
        if self.test:
            index = index + self.num_data
            
        content_path = self.content_paths[index]
        content_img = Image.open(content_path).convert("RGB")
        content_img = self.transform(content_img)
        
        # s_index = random.randint(0, len(self.style_paths)-1)
        s_index = index
        style_path = self.style_paths[s_index]
        style_img = Image.open(style_path).convert("RGB")
        style_img = self.transform(style_img)
        return content_img, style_img
    

def get_dataloader(root, transform=None, batch_size=4, shuffle=False, num_data=None, test=False):
    dataset = ImageDataset(root, transform, num_data, test)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader
