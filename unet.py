import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, inchannel,outchannel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True)
        
        ) 
        

    def forward(self,x):
        return self.conv(x)
         
class UNet(nn.Module):
    
    def __init__(self,inchannel,outchannel,features = [64,128,256,512]):
        
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #down

        for feature in features:
            self.downs.append(DoubleConv(inchannel,feature))
            inchannel = feature


        #up

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,2,2)
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], outchannel, kernel_size=1)          


    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]


            if x.shape[2:] != skip_connection.shape[2:]:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(inchannel=1, outchannel=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()