import torch 
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision

from torch import Tensor

class Refinement(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dense_net = torchvision.models.densenet161(pretrained = True)
        # num_out_channels = self.dense_net.num_init_features
        num_out_channels = 96
        print(num_out_channels)
        
        self.image_forward = nn.Sequential(
            nn.Conv2d(3, num_out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.anomap_forward = nn.Sequential(
            nn.Conv2d(1, num_out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.dense_net.features.conv0 = nn.Identity()
        # self.dense_net.features.conv0 = torch.cat((self.image_forward, self.anomap_forward), dim = 1)
        
    def forward(self, image, ano_map):       
        image = self.image_forward(image)
        print(ano_map.shape)
        ano_map = self.anomap_forward(ano_map)
        x = torch.cat((image, ano_map), dim = 1)
        self.dense_net(x)
        return x
    
    def loss_fn(self, x, y): 
            print(x.shape[-2:])
            l_reg = torch.linalg.norm(x - y) / (x.shape[-2:][0] * x.shape[-2:][1])
            l_grad = (torch.linalg.norm(torch.gradient(x, dim = 3) - torch.gradient(y, dim = 3)) 
                      + torch.linalg.norm(torch.gradient(x, dim = 2) - torch.gradient(y, dim = 2))) / (2 * (x.shape[-2:][0] * x.shape[-2:][1]) )
            return (l_grad + l_reg) / 2
        
    def training_step(self, batch: Tensor, batch_idx):
        x, y = batch['data'], batch['label']
        pred  = self(x)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        x, y = batch['data'], batch['label']
        pred  = self(x)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])