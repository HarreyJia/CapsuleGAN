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
    # self.decoder = Decoder()
    
    # self.mse_loss = nn.MSELoss()
      
  def forward(self, data):
    output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
    return output

    #reconstructions, masked = self.decoder(output, data)
    #return output, reconstructions, masked
    
  # def loss(self, data, x, target, reconstructions):
  #   return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)
  
  # def margin_loss(self, x, labels, size_average=True):
  #   batch_size = x.size(0)

  #   v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

  #   left = F.relu(0.9 - v_c).view(batch_size, -1)
  #   right = F.relu(v_c - 0.1).view(batch_size, -1)

  #   loss = labels * left + 0.5 * (1.0 - labels) * right
  #   loss = loss.sum(dim=1).mean()

  #   return loss
  
  # def reconstruction_loss(self, data, reconstructions):
  #   loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
  #   return loss * 0.0005


class CapsuleLoss(nn.Module):
  def __init__(self):
      super(CapsuleLoss, self).__init__()
      # self.reconstruction_loss = nn.MSELoss(size_average=False)

  # def forward(self, images, labels, classes, reconstructions):
  def forward(self, classes, labels):
      left = F.relu(0.9 - classes, inplace=True) ** 2
      right = F.relu(classes - 0.1, inplace=True) ** 2

      margin_loss = labels * left + 0.5 * (1. - labels) * right
      margin_loss = margin_loss.sum()

      return margin_loss

      # assert torch.numel(images) == torch.numel(reconstructions)
      # images = images.view(reconstructions.size()[0], -1)
      # reconstruction_loss = self.reconstruction_loss(reconstructions, images)

      # return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


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



# Generator
class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu

    self.project = nn.Sequential(
      nn.Linear(nz, 4 * 4 * ngf * 8, bias=False)
    )
    self.deconv = nn.Sequential(
      # input is Z, going into a deconvolution
      # state size. (ngf*8) x 4 x 4 x 4
      nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm3d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8 x 8
      nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm3d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16 x 16
      nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm3d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32 x 32
      nn.ConvTranspose3d(ngf, 1, kernel_size=4, stride=2, padding=1, bias=False),
      nn.Tanh()
      # state size. 1 x 64 x 64 x 64
    )

  def forward(self, input):
    x = self.project(input)
    # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
    x = x.view(1, ngf * 8, 4, 4, 4)
    x = self.deconv(x)
    return x