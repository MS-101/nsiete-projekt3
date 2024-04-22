from dataset import Selfie2AnimeDataset
from torch.utils.data import DataLoader


class DataModule():
    def __init__(self):
        self.train_dataset = Selfie2AnimeDataset(root_dir='../data/train')
        self.val_dataset = Selfie2AnimeDataset(root_dir='../data/val')
        self.test_dataset = Selfie2AnimeDataset(root_dir='../data/test')
    
        self.batch_size = 8
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
