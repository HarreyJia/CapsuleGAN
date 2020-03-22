from model import CapsNet, CapsuleLoss, Generator

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

USE_CUDA = FALSE

BATCH_SIZE = 25
nz = 100

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")

n = torch.randn(BATCH_SIZE, 1, 28, 28, 28)

netD = CapsNet()
netG = Generator()

# if USE_CUDA:
#   capsule_net = model.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = CapsuleLoss()

fixed_noise = torch.randn(BATCH_SIZE, nz, 1, 1)
real_label = 1
fake_label = 0

n_epochs = 30

for epoch in range(n_epochs):
  for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize Lm(D(x), T = 0) + Lm(D(G(z)), T = 1)
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), fake_label, device=device)

        output = netD(real_cpu)
        errD_real = (output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(real_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize Lm(D(G(z)), T = 1)
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, n_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # if i % 100 == 0:
        #     vutils.save_image(real_cpu,
        #             '%s/real_samples.png' % opt.outf,
        #             normalize=True)
        #     fake = netG(fixed_noise)
        #     vutils.save_image(fake.detach(),
        #             '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
        #             normalize=True)