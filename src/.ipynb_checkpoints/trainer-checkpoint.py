import os
import matplotlib.pyplot as plt
import torch


def decide_device():
    if (torch.cuda.is_available()): return "cuda"
    return "cpu"

class Trainer:
    def __init__(self, config):
        self.device = torch.device(decide_device())

        self.datamodule = config['datamodule']
        
        self.generator_A2B = config['generator_A2B'].to(self.device)
        self.generator_B2A = config['generator_B2A'].to(self.device)
        self.discriminator_A = config['discriminator_A'].to(self.device)
        self.discriminator_B = config['discriminator_B'].to(self.device)
        
        self.criterion_GAN = config['criterion_GAN']
        self.criterion_cycle = config['criterion_cycle']
        self.criterion_identity = config['criterion_identity']
        
        self.optimizer_G = config['optimizer_G']
        self.optimizer_D_A = config['optimizer_D_A']
        self.optimizer_D_B = config['optimizer_D_B']
        
        self.max_epoch = config['max_epoch']
        self.output_dir = config['output_dir']

    def fit(self, checkpoint=None):
        if (checkpoint):
            self.load_checkpoint(filename=checkpoint)
        else:
            self.train_losses_G = []
            self.train_losses_D_A = []
            self.train_losses_D_B = []

            self.val_losses_G = []
            self.val_losses_D_A = []
            self.val_losses_D_B = []

            self.cur_epoch = 0
        
        for epoch in range(self.cur_epoch, self.max_epoch):
            self.cur_epoch = epoch
            
            train_loss_G, train_loss_D_A, train_loss_D_B = self.train_epoch(epoch)
            self.train_losses_G.append(train_loss_G)
            self.train_losses_D_A.append(train_loss_D_A)
            self.train_losses_D_B.append(train_loss_D_B)

            val_loss_G, val_loss_D_A, val_loss_D_B = self.val_epoch(epoch)
            self.val_losses_G.append(val_loss_G)
            self.val_losses_D_A.append(val_loss_D_A)
            self.val_losses_D_B.append(val_loss_D_B)

            self.save_checkpoint(filename=f"epoch_{epoch+1}.pt")

        self.plot_loss(filename='generator_loss.png', caption='Generator loss', train_losses=self.train_losses_G, val_losses=self.val_losses_G)
        self.plot_loss(filename='discriminator_A_loss.png', caption='Discriminator A loss', train_losses=self.train_losses_D_A, val_losses=self.val_losses_D_A)
        self.plot_loss(filename='discriminator_B_loss.png', caption='Discriminator B loss', train_losses=self.train_losses_D_B, val_losses=self.val_losses_D_B)
          
    def train_epoch(self, epoch):
        dataloader = self.datamodule.train_loader
    
        self.generator_A2B.train()
        self.generator_B2A.train()
        self.discriminator_A.train()
        self.discriminator_B.train()
    
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0
    
        for real_A, real_B in dataloader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
    
            # Determine target dimensions and set them to the appropriate size
            target_shape = real_A.size(0)  # Using batch size as the reference
            target_real = torch.ones(target_shape, device=self.device)
            target_fake = torch.zeros(target_shape, device=self.device)
    
            # TRAIN GENERATORS A2B, B2A
    
            self.optimizer_G.zero_grad()
    
            # identity loss
            
            identity_A = self.generator_B2A(real_A)
            loss_identity_A = self.criterion_identity(identity_A, real_A) * 5.0
            
            identity_B = self.generator_A2B(real_B)
            loss_identity_B = self.criterion_identity(identity_B, real_B) * 5.0
    
            # GAN loss
            
            fake_A = self.generator_B2A(real_B)
            pred_fake = self.discriminator_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)
    
            fake_B = self.generator_A2B(real_A)
            pred_fake = self.discriminator_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)
    
            # cycle loss
    
            recovered_A = self.generator_B2A(fake_B)
            loss_cycle_B2A = self.criterion_cycle(recovered_A, real_A) * 10.0
    
            recovered_B = self.generator_A2B(fake_A)
            loss_cycle_A2B = self.criterion_cycle(recovered_B, real_B) * 10.0
    
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B + loss_cycle_B2A
            total_loss_G += loss_G.item()
            
            loss_G.backward()
            self.optimizer_G.step()
    
            # TRAIN DISCRIMINATOR A
    
            self.optimizer_D_A.zero_grad()
    
            # real loss
    
            pred_real = self.discriminator_A(real_A)
            loss_D_A_real = self.criterion_GAN(pred_real, target_real)
    
            # fake loss
    
            pred_fake = self.discriminator_A(fake_A.detach())
            loss_D_A_fake = self.criterion_GAN(pred_fake, target_fake)
            
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            total_loss_D_A += loss_D_A.item()
        
            loss_D_A.backward()
            self.optimizer_D_A.step()
    
            # TRAIN DISCRIMINATOR B
    
            self.optimizer_D_B.zero_grad()
    
            # real loss
    
            pred_real = self.discriminator_B(real_B)
            loss_D_B_real = self.criterion_GAN(pred_real, target_real)
    
            # fake loss
    
            pred_fake = self.discriminator_B(fake_B.detach())
            loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)
    
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            total_loss_D_B += loss_D_B.item()
        
            loss_D_B.backward()
            self.optimizer_D_B.step()
    
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D_A = total_loss_D_A / len(dataloader)
        avg_loss_D_B = total_loss_D_B / len(dataloader)
        
        print(f'Epoch {epoch + 1}: Training loss: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')
    
        return avg_loss_G, avg_loss_D_A, avg_loss_D_B


    def val_epoch(self, epoch):
        dataloader = self.datamodule.val_loader
    
        self.generator_A2B.eval()
        self.generator_B2A.eval()
        self.discriminator_A.eval()
        self.discriminator_B.eval()
    
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0
    
        for real_A, real_B in dataloader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
    
            # Determine target dimensions and set them to the appropriate size
            target_shape = real_A.size(0)  # Using batch size as the reference
            target_real = torch.ones(target_shape, device=self.device)
            target_fake = torch.zeros(target_shape, device=self.device)
    
            # Evaluate generators A2B and B2A
            
            # Identity loss
            identity_A = self.generator_B2A(real_A)
            loss_identity_A = self.criterion_identity(identity_A, real_A) * 5.0
            
            identity_B = self.generator_A2B(real_B)
            loss_identity_B = self.criterion_identity(identity_B, real_B) * 5.0
    
            # GAN loss
            fake_A = self.generator_B2A(real_B)
            pred_fake = self.discriminator_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)
    
            fake_B = self.generator_A2B(real_A)
            pred_fake = self.discriminator_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)
    
            # Cycle loss
            recovered_A = self.generator_B2A(fake_B)
            loss_cycle_B2A = self.criterion_cycle(recovered_A, real_A) * 10.0
    
            recovered_B = self.generator_A2B(fake_A)
            loss_cycle_A2B = self.criterion_cycle(recovered_B, real_B) * 10.0
    
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B + loss_cycle_B2A
            total_loss_G += loss_G.item()
    
            # Evaluate discriminator A
            pred_real = self.discriminator_A(real_A)
            loss_D_A_real = self.criterion_GAN(pred_real, target_real)
    
            pred_fake = self.discriminator_A(fake_A.detach())
            loss_D_A_fake = self.criterion_GAN(pred_fake, target_fake)
    
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            total_loss_D_A += loss_D_A.item()
    
            # Evaluate discriminator B
            pred_real = self.discriminator_B(real_B)
            loss_D_B_real = self.criterion_GAN(pred_real, target_real)
    
            pred_fake = self.discriminator_B(fake_B.detach())
            loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)
    
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            total_loss_D_B += loss_D_B.item()
    
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D_A = total_loss_D_A / len(dataloader)
        avg_loss_D_B = total_loss_D_B / len(dataloader)
        
        print(f'Epoch {epoch + 1}: Validation loss: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')
    
        return avg_loss_G, avg_loss_D_A, avg_loss_D_B

    def load_checkpoint(self, filename):
        filename = os.path.join(self.output_dir, filename)
        
        checkpoint = torch.load(filename)
        
        self.generator_A2B.load_state_dict(checkpoint['generator_A2B'])
        self.generator_B2A.load_state_dict(checkpoint['generator_B2A'])
        self.discriminator_A.load_state_dict(checkpoint['discriminator_A'])
        self.discriminator_B.load_state_dict(checkpoint['discriminator_B'])
        
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])

        self.train_losses_G = checkpoint['train_losses_G']
        self.train_losses_D_A = checkpoint['train_losses_D_A']
        self.train_losses_D_B = checkpoint['train_losses_D_B']

        self.val_losses_G = checkpoint['val_losses_G']
        self.val_losses_D_A = checkpoint['val_losses_D_A']
        self.val_losses_D_B = checkpoint['val_losses_D_B']

        self.cur_epoch = checkpoint['cur_epoch']
    
    def save_checkpoint(self, filename):
        filename = os.path.join(self.output_dir, filename)
        
        torch.save({
            'generator_A2B': self.generator_A2B.state_dict(),
            'generator_B2A': self.generator_B2A.state_dict(),
            'discriminator_A': self.discriminator_A.state_dict(),
            'discriminator_B': self.discriminator_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
            'train_losses_G': self.train_losses_G,
            'train_losses_D_A': self.train_losses_D_A,
            'train_losses_D_B': self.train_losses_D_B,
            'val_losses_G': self.val_losses_G,
            'val_losses_D_A': self.val_losses_D_A,
            'val_losses_D_B': self.val_losses_D_B,
            'cur_epoch': self.cur_epoch
        }, filename)

    def plot_loss(self, filename, caption, train_losses, val_losses):
        filename = os.path.join(self.output_dir, filename)

        plt.clf()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(caption)
        plt.legend()
        plt.savefig(filename)
