import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms

USE_CUDA = FALSE

def softmax(input, dim=1):
  transposed_input = input.transpose(dim, len(input.size()) - 1)
  softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
  return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class ConvLayer(nn.Module):
  def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
    super(ConvLayer, self).__init__()

    self.conv = nn.Conv3d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=1
                          )

  def forward(self, x):
    return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
  def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6 * 6):
    super(PrimaryCaps, self).__init__()

    self.capsules = nn.ModuleList([
      nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) 
                  for _ in range(num_capsules)])
  
  def forward(self, x):
    u = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
    u = torch.cat(u, dim=-1)
    return self.squash(u)
  
  def squash(self, input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor



class DigitCaps(nn.Module):
  def __init__(self, num_capsules=10, num_routes=32 * 6 * 6 * 6, in_channels=8, out_channels=16, num_iterations=3):
    super(DigitCaps, self).__init__()

    self.in_channels = in_channels
    self.num_routes = num_routes
    self.num_capsules = num_capsules
    self.num_iterations = num_iterations
    self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

  def forward(self, x):
    # 矩阵相乘
    # x.size(): [1, batch_size, in_capsules, 1, dim_in_capsule]
    # weight.size(): [num_capsules, 1, num_route, in_channels, out_channels]
    priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

    print()
    print(x[None, :, :, None, :].size())
    print(self.route_weights[:, None, :, :, :].size())
    print(priors.size())
    print()

    # logits = Variable(torch.zeros(*priors.size())).cuda()
    logits = Variable(torch.zeros(*priors.size()))
    for i in range(self.num_iterations):
      probs = softmax(logits, dim=2)
      outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

      if i != self.num_routes - 1:
        delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
        logits = logits + delta_logits
    
    return outputs
  
  def squash(self, input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.reconstraction_layers = nn.Sequential(
      nn.Linear(16 * 10, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 1024),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 784),
      nn.Sigmoid()
    )
      
  def forward(self, x, data):
    classes = (x ** 2).sum(dim=-1) ** 0.5
    classes = F.softmax(classes, dim=-1)
    
    _, max_length_indices = classes.max(dim=1)
    masked = Variable(torch.sparse.torch.eye(10))
    if USE_CUDA:
      masked = masked.cuda()
    masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
    
    reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
    reconstructions = reconstructions.view(-1, 1, 28, 28)
    
    return reconstructions, masked


class CapsNet(nn.Module):
  def __init__(self):
    super(CapsNet, self).__init__()
    self.conv_layer = ConvLayer()
    self.primary_capsules = PrimaryCaps()
    self.digit_capsules = DigitCaps().squeeze().transpose(0, 1)
    self.decoder = Decoder()
    
    self.mse_loss = nn.MSELoss()
      
  def forward(self, data):
    output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))

    reconstructions, masked = self.decoder(output, data)
    return output, reconstructions, masked
    
  def loss(self, data, x, target, reconstructions):
    return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)
  
  def margin_loss(self, x, labels, size_average=True):
    batch_size = x.size(0)

    v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

    left = F.relu(0.9 - v_c).view(batch_size, -1)
    right = F.relu(v_c - 0.1).view(batch_size, -1)

    loss = labels * left + 0.5 * (1.0 - labels) * right
    loss = loss.sum(dim=1).mean()

    return loss
  
  def reconstruction_loss(self, data, reconstructions):
    loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
    return loss * 0.0005


n = torch.randn(25, 1, 28, 28, 28)

model = CapsNet()

if USE_CUDA:
  capsule_net = capsule_net.cuda()

optimizer = Adam(capsule_net.parameters())


# Training
batch_size = 100
mnist = Mnist(batch_size)

n_epochs = 30

for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    for batch_id, (data, target) in enumerate(mnist.train_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        
        if batch_id % 100 == 0:
            print("train accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
        
    print(train_loss / len(mnist.train_loader))
        
    capsule_net.eval()
    test_loss = 0
    for batch_id, (data, target) in enumerate(mnist.test_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data[0]
        
        if batch_id % 100 == 0:
            print("test accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
    
    print(test_loss / len(mnist.test_loader))


# conv = ConvLayer()
# conv_result = conv(n)
# print(conv_result.size()) # torch.Size([25, 256, 20, 20, 20])

# primary_capsules = PrimaryCaps()
# prim_result = primary_capsules(conv_result)
# print(prim_result.size()) # torch.Size([25, 6912, 8])

# digit_capsules = DigitCaps()
# digit_result = digit_capsules(prim_result)
# print(digit_result.size()) # torch.Size([10, 25, 1, 1, 16])
# digit_result = digit_result.squeeze()
# print(digit_result.size()) # torch.Size([10, 25, 16])
# digit_result = digit_result.transpose(0, 1)
# print(digit_result.size()) # torch.Size([25, 10, 16])