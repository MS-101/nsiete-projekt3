import torch


def decide_device():
    if (torch.cuda.is_available()): return "cuda"
    return "cpu"

class Trainer:
    def __init__(self, datamodule, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
            criterion_GAN, criterion_cycle, criterion_identity, optimizer_G, optimizer_D_A, optimizer_D_B,
            epochs, output_dir):
        self.device = torch.device(decide_device())

        self.datamodule = datamodule
        self.generator_A2B = generator_A2B.to(self.device)
        self.generator_B2A = generator_B2A.to(self.device)
        self.discriminator_A = discriminator_A.to(self.device)
        self.discriminator_B = discriminator_B.to(self.device)
        self.criterion_GAN = criterion_GAN
        self.criterion_cycle = criterion_cycle
        self.criterion_identity = criterion_identity
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.epochs = epochs
        self.output_dir = output_dir

    def fit(self):
        train_losses_G = []
        train_losses_D_A = []
        train_losses_D_B = []
        
        val_losses_G = []
        val_losses_D_A = []
        val_losses_D_B = []

        for epoch in range(self.epochs):
            train_loss_G, train_loss_D_A, train_loss_D_B = self.train_epoch(epoch)
            train_losses_G.append(train_loss_G)
            train_losses_D_A.append(train_loss_D_A)
            train_losses_D_B.append(train_loss_D_B)

            val_loss_G, val_loss_D_A, val_loss_D_B = self.val_epoch(epoch)
            val_losses_G.append(val_loss_G)
            val_losses_D_A.append(val_loss_D_A)
            val_losses_D_B.append(val_loss_D_B)

        self.plot_loss(filename='generator_loss.png', caption='Generator loss', train_losses=train_losses_G, val_losses=val_losses_G)
        self.plot_loss(filename='discriminator_A_loss.png', caption='Discriminator A loss', train_losses=train_losses_D_A, val_losses=val_losses_D_A)
        self.plot_loss(filename='discriminator_B_loss.png', caption='Discriminator B loss', train_losses=train_losses_D_B, val_losses=val_losses_D_B)
          
    def train_epoch(self, epoch):
        dataloader = self.datamodule.train_loader

        self.model.train()
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0

        target_dims = (3, 64, 64)
        target_real = torch.full(target_dims, 1)
        target_fake = torch.full(target_dims, 0)

        for real_A, real_B in dataloader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

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
            loss_cycle_B2A = self.cricerion_cycle(recovered_A, real_A) * 10.0

            recovered_B = self.generator_A2B(fake_A)
            loss_cycle_A2B = self.cricerion_cycle(recovered_B, real_B) * 10.0

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

            pred_fake = self.discriminator_B(fake_A)
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

            pred_fake = self.discriminator_B(fake_B)
            loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            total_loss_D_B += loss_D_B.item()
        
            loss_D_B.backward()
            self.optimizer_D_B.step()

        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D_A = total_loss_D_A / len(dataloader)
        avg_loss_D_B = total_loss_D_B / len(dataloader)
        
        print(f'Epoch {epoch+1}: Training loss: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')

        return avg_loss_G, avg_loss_D_A, avg_loss_D_B

    def val_epoch(self, epoch):
        dataloader = self.datamodule.val_loader

        self.model.eval()
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0

        target_dims = (3, 64, 64)
        target_real = torch.full(target_dims, 1)
        target_fake = torch.full(target_dims, 0)
        
        for real_A, real_B in dataloader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # TRAIN GENERATORS A2B, B2A
            
            self.optimizer_G.zero_grad()

            # identity loss
            
            identity_A = self.generator_B2A(real_A)
            loss_identity_A = self.criterion_identity(identity_A, real_A)*5.0
            
            identity_B = self.generator_A2B(real_B)
            loss_identity_B = self.criterion_identity(identity_B, real_B)*5.0

            # GAN loss
            
            fake_A = self.generator_B2A(real_B)
            pred_fake = self.discriminator_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

            fake_B = self.generator_A2B(real_A)
            pred_fake = self.discriminator_B(fake_B)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_fake)

            # cycle loss

            recovered_A = self.generator_B2A(fake_B)
            loss_cycle_ABA = self.cricerion_cycle(recovered_A, real_A) * 10.0

            recovered_B = self.generator_A2B(fake_A)
            loss_cycle_BAB = self.cricerion_cycle(recovered_B, real_B) * 10.0

            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            total_loss_G += loss_G.item()

            # TRAIN DISCRIMINATOR A

            self.optimizer_D_A.zero_grad()

            # real loss

            pred_real = self.discriminator_A(real_A)
            loss_D_A_real = self.criterion_GAN(pred_real, target_real)

            # fake loss

            pred_fake = self.discriminator_A(fake_A)
            loss_D_A_fake = self.criterion_GAN(pred_fake, target_fake)
            
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            total_loss_D_A += loss_D_A.item()
        
            # TRAIN DISCRIMINATOR B

            self.optimizer_D_B.zero_grad()

            # real loss

            pred_real = self.discriminator_B(real_B)
            loss_D_B_real = self.criterion_GAN(pred_real, target_real)

            # fake loss

            pred_fake = self.discriminator_B(fake_B)
            loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            total_loss_D_B += loss_D_B.item()

        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D_A = total_loss_D_A / len(dataloader)
        avg_loss_D_B = total_loss_D_B / len(dataloader)
        
        print(f'Epoch {epoch+1}: Validation loss: generator = {avg_loss_G}; discriminator A = {avg_loss_D_A}; discriminator B = {avg_loss_D_B}')

        return avg_loss_G, avg_loss_D_A, avg_loss_D_B

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
