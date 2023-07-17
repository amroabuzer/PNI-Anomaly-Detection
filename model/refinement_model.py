import torch 
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F

from torch import Tensor

class Up(nn.Sequential):
    def __init__(self, num_input_channels, num_output_channels):
        super(Up, self).__init__()
        self.convA = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_up = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        x_convA = self.relu(self.convA(x_up))
        x_convB = self.relu(self.convB(x_convA))

        return x_convB


class Refinement(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dense_net = torchvision.models.densenet161(num_classes = 1000, pretrained = True)
        # num_out_channels = self.dense_net.num_init_features
        num_out_channels = 96
        print(num_out_channels)
        
        self.image_forward = nn.Sequential(
            nn.Conv2d(3, num_out_channels//2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.anomap_forward = nn.Sequential(
            nn.Conv2d(1, num_out_channels//2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.initial_layer = nn.Conv2d(
            num_out_channels, self.dense_net.features.conv0.out_channels,
            kernel_size=self.dense_net.features.conv0.kernel_size,
            stride=self.dense_net.features.conv0.stride,
            padding=self.dense_net.features.conv0.padding,
            bias=False
        )

        # Create a new module that combines the concatenated layer with the remaining layers
        self.model = nn.Sequential(*list(self.dense_net.features.children())[1:])
        
        
        num_channels_d32_in = 2208
        num_channels_d32_out = 1000
        
        self.conv_d32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1 = Up(num_input_channels=num_channels_d32_out // 1, num_output_channels=num_channels_d32_out // 2)
        self.up2 = Up(num_input_channels=num_channels_d32_out // 2, num_output_channels=num_channels_d32_out // 4)
        self.up3 = Up(num_input_channels=num_channels_d32_out // 4, num_output_channels=num_channels_d32_out // 8)
        self.up4 = Up(num_input_channels=num_channels_d32_out // 8, num_output_channels=num_channels_d32_out // 16)
        self.up5 = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)
        self.conv3 = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

        
        self.decoder = nn.Sequential(
            
        )
        
    def forward(self, image, ano_map):
        image = self.image_forward(image)
        ano_map = self.anomap_forward(ano_map)
        x = torch.cat((image, ano_map), dim=1)
        x = self.model(x)
        decoder_in = self.conv_d32(x)

        decoder_x2 = self.up1(decoder_in)
        decoder_x4 = self.up2(decoder_x2)
        decoder_x8 = self.up3(decoder_x4)
        decoder_x16 = self.up4(decoder_x8)
        decoder_x32 = self.up5(decoder_x16)
        output = self.conv3(decoder_x32)
        return output
    
    def loss_fn(self, x, y): 
            l_reg = torch.linalg.norm(x - y) / (x.shape[-2:][0] * x.shape[-2:][1])
            l_grad = (torch.linalg.norm((torch.gradient(x, dim = 3)[0] - torch.gradient(y, dim = 3)[0]).squeeze()) 
                      + torch.linalg.norm(torch.gradient(x, dim = 2)[0] - torch.gradient(y, dim = 2)[0])) / (2 * (x.shape[-2:][0] * x.shape[-2:][1]) )
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