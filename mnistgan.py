from __future__ import print_function
import argparse
import os
import sys
import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import numpy as np
from adabound import AdaBound



# add functionality to load models later
# path_netD = os.getcwd()+os.sep+'models'+os.sep+'pytorch'+os.sep+'netD'
# path_netG = os.getcwd()+os.sep+'models'+os.sep+'pytorch'


### setting up some constants
manualSeed = 77
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True
imageSize = 64
niter = 100
beta1 = 0.9
beta2 = 0.999
ngpu = 1
nz = 100 # latent space dims
ngf = 32
ndf = 32
nc = 1 #number of channels = 1 for MNIST
bz=128
optimnames=['sgd']

if torch.cuda.is_available():
    cuda = True
    print("Using cuda")
else:
    cuda = False
    print("Not using cuda")
device = torch.device("cuda:0" if cuda else "cpu")


### loading the data
dataset = dset.MNIST(root='../data', download=True,
                       transform=transforms.Compose([
                           transforms.Resize(imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size= bz,
                                         shuffle=True)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0001, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.001)

def drawProgressBar(percent, barLen = 50):

    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output



# if os.path.isfile(path_netG):
#     netG.load_state_dict(torch.load(path_netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


for optimname in optimnames:
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    #netG = torch.nn.DataParallel(netG)
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    #netD = torch.nn.DataParallel(netD)
    # if opt.netD != '':
    #     netD.load_state_dict(torch.load(opt.netD))
    # print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(bz, nz, 1, 1, device=device)  # 64
    real_label = 1
    fake_label = 0

    # setup optimizer
    if(optimname=='sgd'):
        print('1')
        optimizerD = optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
        optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, beta2))
    elif(optimname=='adam'):
        optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(beta1, beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, beta2))
    else:
        print('aaaa')
        optimizerD = AdaBound(netD.parameters(), lr=0.00001, betas=(beta1, beta2),
                        final_lr=0.0005, gamma=0.001,weight_decay=5e-4)
        optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, beta2))

    start = time.time()
    for epoch in range(1, niter + 1):
        counter = 0
        print("\nEpoch[{}/{}]:\n".format(epoch, niter))
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            #print('1:',errD_real.data.item())
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            #print('1:', errD_real.data.item())
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # drawProgressBar((counter+1)/len(dataloader))
            counter += 1
            if i % 100 == 0:
                vutils.save_image(real_cpu, 'gan_results' + os.sep + str(epoch) +optimname+ 'real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  'gan_results' + os.sep + str(epoch) +optimname+ '_fake_samples_epoch_{}.png'.format(epoch),
                                  normalize=True)
        print('\n[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'.format(epoch,
                                                                                                     niter,
                                                                                                     errD.item(),
                                                                                                     errG.item(),
                                                                                                     D_x,
                                                                                                     D_G_z1,
                                                                                                     D_G_z2))
    end = time.time()
    times = end - start
    times = np.array([times], np.float32)
    np.save(os.path.join('times', optimname + '_times.npy'), times)
    print(times)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('gan_results', epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('gan_results', epoch))

