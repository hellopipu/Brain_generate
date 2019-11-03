import torch
import torch.nn as nn
from torch.utils import data
from dataset import Dataset_brain
from model import Encoder, Decoder,VAE
import numpy as np
import time
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import seed_torch, gradient_penalty
import torchvision.utils as vutils
import matplotlib.animation as animation

#seed_torch()

file_path = '/home/xin/PycharmProjects/GAN_repo/brats18_dataset/npy_gan/gan_t2.npy'
train_data = Dataset_brain(file_path)
batch_size = 64

train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
len_loader = len(train_loader)
# gen = Decoder()
# dis = Encoder()
vae=VAE()
# optimizer_g = torch.optim.Adam(gen.parameters(), lr=0.0002)  # can't be trained without wgan-gp on this lr
# optimizer_d = torch.optim.Adam(dis.parameters(), lr=0.0002)
# params= [gen,dis]
# pp=[ mlp.parameters() for mlp in params]
# print(params)
optimizer = torch.optim.Adam( vae.parameters(), lr=0.0001)  # 0.0002 [{"params":gen.parameters()},{"params":dis.parameters()}]
# print(optimizer)
# optimizer_d = torch.optim.Adam(dis.parameters(), lr=0.00002)
# gen.cuda()
# dis.cuda()
vae.cuda()
EPOCH = 50
KL_LOSS = []
RE_LOSS = []
LOSS = []
img_list = []
# loss_=0
# recon_loss_ = 0
# kl_loss_ = 0
f = open("./loss_gan.txt", 'a')
print(time.strftime('|---------%Y-%m-%d   %H:%M:%S---------| dcgan', time.localtime(time.time())), file=f)
fixed_noise = torch.randn(64, 100, 1, 1).cuda()
criterion = nn.BCELoss()

for epoch in range(EPOCH):
    num_iter = 0
    loss_ = 0
    recon_loss_ = 0
    kl_loss_ = 0
    vae.train()

    for i, real in enumerate(train_loader):


        o_mu, o_var,fake = vae(real.float().cuda())
       # if i==0:

        #    print(fake[0].max(),real[0].max())

         #   plt.imshow(fake[0,0].cpu().data)
         #   plt.show()


        recon_loss = 128*128*criterion(fake,real.float().cuda())
        # print(recon_loss)
        kl_loss = torch.mean(0.5*torch.sum(torch.exp(o_var)+o_mu**2-1.-o_var,1))
        loss = recon_loss + kl_loss
        # print('ok1')
        optimizer.zero_grad()
        # print('ok2')
        loss.backward() #retain_graph=True
        optimizer.step()


        recon_loss_ += recon_loss.item() #+ kl_loss.item()
        kl_loss_ += kl_loss.item()
        loss_+=loss.item()
        # num_iter += real.size(0)
    # print('recon', recon_loss.item(), 'kl', kl_loss.item(), 'loss', loss.item())
    RE_LOSS.append(recon_loss_ / len_loader)
    KL_LOSS.append(kl_loss_ / len_loader)
    LOSS.append(loss_ / len_loader)

    with torch.no_grad():
        vae.eval()
        ff = vae.dec(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(ff, padding=2, normalize=True))

    print('EPOCH %d : recon_loss : %.4f , kl_loss = %.4f , loss = %.4f ' % (epoch, recon_loss_ / len_loader, kl_loss_ / len_loader, loss_ / len_loader))
    ## save fig
    plt.axis('off')
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('glips.png', format='png')
    plt.close()

    ### save plot
    x = [i for i in range(epoch + 1)]
    plt.plot(x, RE_LOSS, label='RE_LOSS')
    plt.plot(x, KL_LOSS, label='KL_LOSS')
    plt.plot(x, LOSS, label='LOSS')
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
torch.save(vae.dec.state_dict(), 'gen_.pth')
torch.save(vae.enc.state_dict(), 'dis_.pth')

