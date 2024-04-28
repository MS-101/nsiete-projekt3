import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image


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

        self.save_plot(filename='generator_loss.png', caption='Generator loss', metric_name='Loss', train_values=self.train_losses_G, val_values=self.val_losses_G)
        self.save_plot(filename='discriminator_A_loss.png', caption='Discriminator A loss', metric_name='Loss', train_values=self.train_losses_D_A, val_values=self.val_losses_D_A)
        self.save_plot(filename='discriminator_B_loss.png', caption='Discriminator B loss', metric_name='Loss', train_values=self.train_losses_D_B, val_values=self.val_losses_D_B)

        self.test_epoch()
          
    def train_epoch(self, epoch):
        avg_loss_G, avg_loss_D_A, avg_loss_D_B = self.epoch(dataloader=self.datamodule.train_loader, save_images=False)

        print(f'Epoch {epoch + 1} training: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')
    
        return avg_loss_G, avg_loss_D_A, avg_loss_D_B


    def val_epoch(self, epoch):
        with torch.no_grad():
            avg_loss_G, avg_loss_D_A, avg_loss_D_B = self.epoch(dataloader=self.datamodule.val_loader, save_images=False)

        print(f'Epoch {epoch + 1} validation: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')
    
        return avg_loss_G, avg_loss_D_A, avg_loss_D_B

    def test_epoch(self):
        with torch.no_grad():
            avg_loss_G, avg_loss_D_A, avg_loss_D_B = self.epoch(dataloader=self.datamodule.test_loader, save_images=True)

        print(f'Testing: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')
    
    def epoch(self, dataloader, save_images):
        batch_size = dataloader.batch_size

        if (torch.is_grad_enabled()):
            self.generator_A2B.train()
            self.generator_B2A.train()
            self.discriminator_A.train()
            self.discriminator_B.train()
        else:
            self.generator_A2B.eval()
            self.generator_B2A.eval()
            self.discriminator_A.eval()
            self.discriminator_B.eval()

        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0

        for cur_batch, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
        
            target_shape = (real_A.size(0), 1)
            target_real = torch.ones(target_shape, device=self.device)
            target_fake = torch.zeros(target_shape, device=self.device)
    
            # GENERATORS
    
            if (torch.is_grad_enabled()):
                self.optimizer_G.zero_grad()
    
            # identity loss
            print(real_A.size())
            identity_A = self.generator_B2A(real_A)
            loss_identity_A = self.criterion_identity(identity_A, real_A) * 5.0

            if (save_images):
                for i in range(min(batch_size, len(real_A))):
                    self.save_image(filename=f'identity_A_{cur_batch*batch_size+i}.png', origin=real_A[i], result=identity_A[i])
            
            identity_B = self.generator_A2B(real_B)
            loss_identity_B = self.criterion_identity(identity_B, real_B) * 5.0

            if (save_images):
                for i in range(min(batch_size, len(real_B))):
                    self.save_image(filename=f'identity_B_{cur_batch*batch_size+i}.png', origin=real_B[i], result=identity_B[i])
    
            # GAN loss
            
            fake_A = self.generator_B2A(real_B)
            pred_fake = self.discriminator_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

            if (save_images):
                for i in range(min(batch_size, len(real_B))):
                    self.save_image(filename=f'fake_A_{cur_batch*batch_size+i}.png', origin=real_B[i], result=fake_A[i])
    
            fake_B = self.generator_A2B(real_A)
            pred_fake = self.discriminator_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

            if (save_images):
                for i in range(min(batch_size, len(real_A))):
                    self.save_image(filename=f'fake_B_{cur_batch*batch_size+i}.png', origin=real_A[i], result=fake_B[i])
    
            # cycle loss
    
            recovered_A = self.generator_B2A(fake_B)
            loss_cycle_B2A = self.criterion_cycle(recovered_A, real_A) * 10.0

            if (save_images):
                for i in range(min(batch_size, len(fake_B))):
                    self.save_image(filename=f'recovered_A_{cur_batch*batch_size+i}.png', origin=fake_B[i], result=recovered_A[i])
    
            recovered_B = self.generator_A2B(fake_A)
            loss_cycle_A2B = self.criterion_cycle(recovered_B, real_B) * 10.0

            if (save_images):
                for i in range(min(batch_size, len(fake_A))):
                    self.save_image(filename=f'recovered_B_{cur_batch*batch_size+i}.png', origin=fake_A[i], result=recovered_B[i])

            # total loss
    
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B + loss_cycle_B2A
            total_loss_G += loss_G.item()
            
            if (torch.is_grad_enabled()):
                loss_G.backward()
                self.optimizer_G.step()
    
            # DISCRIMINATOR A
    
            if (torch.is_grad_enabled()):
                self.optimizer_D_A.zero_grad()
    
            # real loss
    
            pred_real = self.discriminator_A(real_A)
            loss_D_A_real = self.criterion_GAN(pred_real, target_real)
    
            # fake loss
    
            pred_fake = self.discriminator_A(fake_A.detach())
            loss_D_A_fake = self.criterion_GAN(pred_fake, target_fake)
            
            # total loss

            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            total_loss_D_A += loss_D_A.item()

            if (torch.is_grad_enabled()):
                loss_D_A.backward()
                self.optimizer_D_A.step()
    
            # DISCRIMINATOR B
    
            if (torch.is_grad_enabled()):
                self.optimizer_D_B.zero_grad()
    
            # real loss
    
            pred_real = self.discriminator_B(real_B)
            loss_D_B_real = self.criterion_GAN(pred_real, target_real)
    
            # fake loss
    
            pred_fake = self.discriminator_B(fake_B.detach())
            loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)
    
            # total loss

            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            total_loss_D_B += loss_D_B.item()

            if (torch.is_grad_enabled()):
                loss_D_B.backward()
                self.optimizer_D_B.step()
    
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D_A = total_loss_D_A / len(dataloader)
        avg_loss_D_B = total_loss_D_B / len(dataloader)
    
        return avg_loss_G, avg_loss_D_A, avg_loss_D_B

    def load_checkpoint(self, filename):
        dir = os.path.join(self.output_dir, 'checkpoints')
        filename = os.path.join(dir, filename)
        
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
        dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(dir, filename)
        
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

    def save_image(self, filename, origin, result):
        dir = os.path.join(self.output_dir, 'images')
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(dir, filename)

        origin_image = F.to_pil_image(origin)
        result_image = F.to_pil_image(result)

        merged_image = Image.new('RGB', (origin_image.width * 2, origin_image.height))
        merged_image.paste(origin_image, (0, 0))
        merged_image.paste(result_image, (origin_image.width, 0))

        merged_image.save(filename)

    def save_plot(self, filename, caption, metric_name, train_values, val_values):
        dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(dir, exist_ok=True)

        filename = os.path.join(dir, filename)

        plt.clf()
        plt.plot(train_values, label='Training')
        plt.plot(val_values, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(caption)
        plt.legend()
        plt.savefig(filename)
