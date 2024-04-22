import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Selfie2AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.domainA_dir = root_dir + '/A'
        self.domainB_dir = root_dir + '/B'
        self.domainA_list = os.listdir(self.domainA_dir)
        self.domainB_list = os.listdir(self.domainB_dir)

    def __len__(self):
        return min(len(self.domainA_list), len(self.domainB_list))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        domainA_path = os.path.join(self.domainA_dir, self.domainA_list[idx])
        domainA_img = Image.open(domainA_path)

        domainB_path = os.path.join(self.domainB_dir, self.domainB_list[idx])
        domainB_img = Image.open(domainB_path)
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.PILToTensor()
        ]) 

        domainA_tensor = transform(domainA_img).to(torch.float32)
        domainB_tensor = transform(domainB_img).to(torch.float32)

        return domainA_tensor, domainB_tensor
