import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_dcsaunet.encoder import CSA
from torchvision.models import mobilenet_v3_small

csa_block = CSA()

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

    
class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(3, channels, kernel_size, padding=  kernel_size // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    
# Define the outline attention module
# class OutlineAttention(nn.Module):
#     batchNorm_momentum = 0.1
#     def __init__(self, channels):
#         super(OutlineAttention, self).__init__()
#         # Define the outline kernel
#         kernel = [[-1, -1, -1],
#                   [-1, 9, -1],
#                   [-1, -1, -1]]
#         # Convert the kernel to a PyTorch tensor
#         kernel = torch.FloatTensor(kernel).expand(channels, channels, 3, 3)
#         # Convert the kernel to a trainable parameter
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#         # Batch normalization layer
#         self.bn = nn.BatchNorm2d(channels, momentum=self.batchNorm_momentum)
#         # ReLU activation
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # Apply the outline convolution
#         x = F.conv2d(x, self.weight, padding=1)
#         # Apply batch normalization
#         x = self.bn(x)
#         # Apply ReLU activation
#         x = self.relu(x)
#         return x
class TAFModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(TAFModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature activation
        self.conv1d = nn.Conv1d(channels, channels // reduction, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(channels // reduction, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),  # Adjusted the output size
        )

    def forward(self, x):
        b, c, h, w = x.size()
      
        # Multipath feature compression
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        x = avg_out + max_out

        # Activation
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add two dimensions for conv2d
        # Convolution 1D
        x_conv1d = self.conv1d(x.squeeze(-1)).squeeze(-1)

        # ReLU after Conv1D
        x_relu_conv1d = self.relu(x_conv1d)

        # Convolution 2D
        x_conv2d = self.conv2d(x_relu_conv1d.unsqueeze(-1).unsqueeze(-1))

        # Apply the fully connected layers
        x_fc = self.fc(x_conv2d.view(b, -1))  # Adjusted the view operation
        # Apply the sigmoid activation
        x_sigmoid = self.sigmoid(x_fc)
        x_sigmoid = x_sigmoid * x_sigmoid
        x_sigmoid = x_sigmoid.view(b, c, 1, 1)  # Add spatial dimensions
        x_sigmoid = x_sigmoid.expand(-1, -1, h, w)  # Expand spatial dimensions

        return x_sigmoid
# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=3, n_classes=1,efficientnet_encoder='b0'):
       super(Model, self).__init__()
       self.pfc = PFC(64)
       self.img_channels = img_channels
       self.n_classes = n_classes
       self.maxpool = nn.MaxPool2d(kernel_size=2)
       self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
       self.up_conv1 = Up()
       self.up_conv2 = Up()
       self.up_conv3 = Up()
       self.up_conv4 = Up()
       self.down1 = csa_block.layer1
       self.down2 = csa_block.layer2
       self.down3 = csa_block.layer3
       self.down4 = csa_block.layer4
       self.up1 = csa_block.layer5
       self.up2 = csa_block.layer6
       self.up3 = csa_block.layer7
       self.up4 = csa_block.layer8
       self.mobilenet_encoder = nn.Sequential(*list(mobilenet_v3_small(pretrained=True).features.children())[:-1])
       #self.outline_attention = OutlineAttention(channels=512)  # Initialize OutlineAttention module
       self.taf_attention = TAFModule(channels=512,reduction = 16)  # Initialize OutlineAttention module


       
    def forward(self, x):
 
        
        x1 = self.mobilenet_encoder(x)
        x1 = self.pfc(x)
    
        x2 = self.maxpool(x1)

        x3 = self.down1(x2)  
        x4 = self.maxpool(x3)
        
        x5 = self.down2(x4)        
        x6 = self.maxpool(x5)
        
        x7 = self.down3(x6)   
        x8 = self.maxpool(x7)
        
        x9 = self.down4(x8)
        #x9 = self.outline_attention(x9)
        x9 = self.taf_attention(x9)


        x10 = self.up_conv1(x9,x7)

        x11 = self.up1(x10)
   
        x12 = self.up_conv2(x11,x5)    
        x13 = self.up2(x12)
       
        x14 = self.up_conv3(x13,x3)   
        x15 = self.up3(x14)
        
        x16 = self.up_conv4(x15,x1)
        x17 = self.up4(x16)
        
        x18 = self.out_conv(x17)
        
        #x19 = torch.sigmoid(x18)
        return x18
