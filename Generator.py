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
