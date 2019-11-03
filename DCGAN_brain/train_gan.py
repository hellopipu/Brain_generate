import torch
import torch.nn as nn
from torch.utils import data
from dataset_brain import Dataset_brain
from model import DCGAN, netD
import numpy as np
import time
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import seed_torch, gradient_penalty
import torchvision.utils as vutils
import matplotlib.animation as animation

seed_torch()

file_path = '/home/xin/PycharmProjects/GAN_repo/brats18_dataset/npy_gan/gan_t2.npy'
train_data = Dataset_brain(file_path)
batch_size = 64

train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

gen = DCGAN()
dis = netD()
# optimizer_g = torch.optim.Adam(gen.parameters(), lr=0.0002)  # can't be trained without wgan-gp on this lr
# optimizer_d = torch.optim.Adam(dis.parameters(), lr=0.0002)
optimizer_g = torch.optim.Adam(gen.parameters(), lr=0.0001)  # 0.0002
optimizer_d = torch.optim.Adam(dis.parameters(), lr=0.00002)
gen.cuda()
dis.cuda()
EPOCH = 50
D_LOSS = []
G_LOSS = []
img_list = []
f = open("./loss_gan.txt", 'a')
print(time.strftime('|---------%Y-%m-%d   %H:%M:%S---------| dcgan', time.localtime(time.time())), file=f)
fixed_noise = torch.randn(64, 100, 1, 1).cuda()
criterion = nn.BCELoss()
for epoch in range(EPOCH):
    num_iter = 0
    d_loss_ = 0
    g_loss_ = 0
    for i, real in enumerate(train_loader):
        ##discriminator
        ##real sample
        label = torch.full((64,),1).cuda()
        output = dis(real.float().cuda()).view(-1)
        # print(output.shape)
        errD_real = criterion(output,label)
        optimizer_d.zero_grad()
        errD_real.backward() #retain_graph=True
        ##fake sample
        # dis(fake.detach())
        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake = gen(noise)
        label.fill_(0)
        output = dis(fake.detach().cuda()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizer_d.step()

        ##generator
        label.fill_(1)
        output = dis(fake.cuda()).view(-1)
        errG = criterion(output, label)
        optimizer_g.zero_grad()
        errG.backward()
        optimizer_g.step()

        d_loss_ += errD_fake.item() + errD_real.item()
        g_loss_ += errG.item()
        num_iter += real.size(0)
        # print('fake', d_loss_fake.item(), 'real', d_loss_real.item(), 'g : -fake', g_loss.item())
    D_LOSS.append(d_loss_ / num_iter)
    G_LOSS.append(g_loss_ / num_iter)

    with torch.no_grad():
        ff = gen(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(ff, padding=2, normalize=True))

    print('EPOCH %d : d_loss : %.4f , g_loss = %.4f ' % (epoch, d_loss_ / num_iter, g_loss_ / num_iter))
    print('EPOCH %d : d_loss : %.4f , g_loss = %.4f ' % (epoch, d_loss_ / num_iter, g_loss_ / num_iter), file=f)
    ## save fig
    plt.axis('off')
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('glips.png', format='png')
    plt.close()

    ### save plot
    x = [i for i in range(epoch + 1)]
    plt.plot(x, G_LOSS, label='generator')
    plt.plot(x, D_LOSS, label='discriminator')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan.png', format='png')
    plt.close()

# Grab a batch of real images from the dataloader
real_batch = next(iter(train_loader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch.cuda()[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig('real_fake.png', format='png')
plt.close()

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save('gan.gif', writer='imagemagick', fps=10)
torch.save(gen.state_dict(), 'gen_.pth')
torch.save(dis.state_dict(), 'dis_.pth')
