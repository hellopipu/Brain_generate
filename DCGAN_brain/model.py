import torch
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self,in_dim=1,conv_dim=64,out_dim=1):
        super(Unet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_dim,conv_dim,kernel_size=3,stride=2,padding=1), #64
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(conv_dim,conv_dim*2,kernel_size=3,stride=2,padding=1), #32
            nn.BatchNorm2d(conv_dim*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 =nn.Sequential(
            nn.Conv2d(conv_dim*2, conv_dim * 4, kernel_size=3, stride=2, padding=1), #16
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=3, stride=2, padding=1), #8
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=3, stride=2, padding=1),  # 8
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=3, stride=2, padding=1),  # 8
            nn.BatchNorm2d(conv_dim * 16),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(conv_dim * 16, conv_dim * 16, kernel_size=3, stride=2, padding=1),  # 8
            nn.BatchNorm2d(conv_dim * 16),
            nn.ReLU(inplace=True)
        )

        ### deconv
        self.deconv1=nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 16,conv_dim * 16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(conv_dim * 16),
            nn.ReLU(inplace=True)
        )
        self.deconv2=nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 16,conv_dim * 8,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.deconv3=nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 8,conv_dim * 8,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(conv_dim * 1),
            nn.ReLU(inplace=True)
        )
        self.deconv7=nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 1,out_dim ,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        out=self.deconv1(x)
        out=self.deconv2(out)
        out=self.deconv3(out)
        out=self.deconv4(out)
        out=self.deconv5(out)
        out=self.deconv6(out)
        out=self.deconv7(out)
        return x,out

class DCGAN(nn.Module):
    def __init__(self,ngf=64):
        super(DCGAN,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100,ngf*8,4,1,0,bias=0),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*8, ngf * 8, 4, 2 , 1, bias=0),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=0),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=0),
            nn.InstanceNorm2d(ngf ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf , 1, 4, 2, 1, bias=0),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                if isinstance(m,nn.InstanceNorm2d) and m.bias is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.main(x)

class netD(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=5):
        super(netD, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
      #  layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim *= 2
        kernel_size = int(image_size / 2 ** repeat_num)
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()   ##add
#         self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m,nn.InstanceNorm2d) and m.bias is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.main(x)
        # print('h',h.shape)
        out_src = self.sigmoid(self.conv1(h))   ##add
        return out_src

if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.rand((4, 100, 1, 1))
    gen=DCGAN()
    dis = netD()
    y=gen(x)
    k=dis(y)
    print(k.shape)
    print('    Total params: %.2fMB' % (sum(p.numel() for p in dis.parameters()) / (1024.0 * 1024) * 4))
