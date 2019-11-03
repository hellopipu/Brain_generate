import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self,ngf=64):
        super(Decoder,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100,ngf*8,4,1,0,bias=0),
           # nn.InstanceNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*8, ngf * 8, 4, 2 , 1, bias=0),
           # nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=0),
           # nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=0),
           # nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=0),
           # nn.InstanceNorm2d(ngf ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf , 1, 4, 2, 1, bias=0),
            nn.Sigmoid(),
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

class Encoder(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=4):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
      #  layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True)) #Leaky

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
           # layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.ReLU(inplace=True)) #Leaky
            curr_dim *= 2
        kernel_size = int(image_size / 2 ** repeat_num)
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, conv_dim*8, kernel_size=4, stride=1, padding=0, bias=True)
        self.linear=nn.Linear(conv_dim*8*25,200)
       # self.conv2 = nn.Conv2d(curr_dim, 100, kernel_size=4, stride=1, padding=0, bias=True)
        # self.sigmoid = nn.Sigmoid()   ##add
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
        out = self.conv1(h)   ##add
        out = self.linear(out.view(out.size(0),-1))
        #out_var = self.conv2(h)  ##add
        return out[:,0:100], out[:,100::]

class VAE(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=5):
        super(VAE, self).__init__()
        self.enc=Encoder()
        self.dec=Decoder()
    def forward(self, x):
        o_mu,o_var = self.enc(x)
        eps =(o_var.data.new(o_var.size()).normal_())
        #eps = torch.randn(64, 100).cuda()
       # print(o_var.shape)
      #  print(eps[0,0,0,0])
        sample_z = o_mu + torch.exp(o_var / 2) * eps
        sample_z =sample_z.unsqueeze(-1).unsqueeze(-1)
       # print(sample_z.shape)
        fake = self.dec(sample_z)
        return o_mu,o_var,fake


if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.rand((4, 1, 128, 128))
    gen=Encoder()
    dis = Decoder()
    y=gen(x)
    print(y.shape)
    k=dis(y)

    print(k.shape)
    print('    Total params: %.2fMB' % (sum(p.numel() for p in dis.parameters()) / (1024.0 * 1024) * 4))
