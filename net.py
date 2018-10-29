
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, zsize, channels = 1, architecture = None):
        super(VAE, self).__init__()
        d = 128
        self.zsize = zsize
        self.deconv1 = nn.ConvTranspose2d(zsize, d * 2, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 2, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        
        self.b_deconv1_1 = nn.ConvTranspose2d(zsize, d*8, 4, 1, 0)
        self.b_deconv1_1_bn = nn.BatchNorm2d(d*8)
        self.b_deconv2 = nn.ConvTranspose2d(d*8, d*8, 4, 2, 1)
        self.b_deconv2_bn = nn.BatchNorm2d(d*8)
        self.b_deconv3 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.b_deconv3_bn = nn.BatchNorm2d(d*4)
        self.b_deconv4 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.b_deconv4_bn = nn.BatchNorm2d(d*2)
        self.b_deconv5 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.b_deconv5_bn = nn.BatchNorm2d(d)
        self.b_deconv6_1 = nn.ConvTranspose2d(d, channels, 4, (3,5), (1,0), output_padding=(1,1))
        self.b_deconv6_2 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.b_deconv6_2_bn = nn.BatchNorm2d(d//2)
        self.b_deconv7 = nn.ConvTranspose2d(d//2, channels, 4, 2, 1)

        self.conv1 = nn.Conv2d(channels, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4_1 = nn.Conv2d(d * 4, zsize, 4, 1, 0)
        self.conv4_2 = nn.Conv2d(d * 4, zsize, 4, 1, 0)
        
        self.b_conv1_1 = nn.Conv2d(channels, d//2, 4, (3,5), 1)
        self.b_conv1_2 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.b_conv2 = nn.Conv2d(d // 2, d, 4, 2, 1)
        self.b_conv2_bn = nn.BatchNorm2d(d)
        self.b_conv3 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.b_conv3_bn = nn.BatchNorm2d(d*2)
        self.b_conv4 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.b_conv4_bn = nn.BatchNorm2d(d*4)
        self.b_conv5 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.b_conv5_bn = nn.BatchNorm2d(d*8)
        self.b_conv6 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.b_conv6_bn = nn.BatchNorm2d(d*8)
        self.b_conv6_1 = nn.Conv2d(d*8, zsize, 4, 1, 0)
        self.b_conv6_2 = nn.Conv2d(d*8, zsize, 4, 1, 0)
        self.b_conv7_1 = nn.Conv2d(d*8, zsize, 4, 1, 0)
        self.b_conv7_2 = nn.Conv2d(d*8, zsize, 4, 1, 0)

        self.architecture = architecture

    def encode(self, x):
        if self.architecture is None:
            x = F.relu(self.conv1(x), 0.2)
            x = F.relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.relu(self.conv3_bn(self.conv3(x)), 0.2)
            h1 = self.conv4_1(x)
            h2 = self.conv4_2(x)
            return h1, h2
        
        elif self.architecture == "b1":
            x = F.relu(self.conv1(x), 0.2)
            x = F.relu(self.b_conv2_bn(self.conv2(x)), 0.2)
            x = F.relu(self.b_conv3_bn(self.conv3(x)), 0.2)
            x = F.relu(self.b_conv4_bn(self.conv4(x)), 0.2)
            x = F.relu(self.b_conv5_bn(self.conv5(x)), 0.2)
            h1 = self.b_conv6_1(x)
            h2 = self.b_conv6_2(x)
            return h1, h2

        elif self.architecture == "b2":
            x = F.relu(self.conv1(x), 0.2)
            x = F.relu(self.b_conv2_bn(self.conv2(x)), 0.2)
            x = F.relu(self.b_conv3_bn(self.conv3(x)), 0.2)
            x = F.relu(self.b_conv4_bn(self.conv4(x)), 0.2)
            x = F.relu(self.b_conv5_bn(self.conv5(x)), 0.2)
            x = F.relu(self.b_conv6_bn(self.conv6(x)), 0.2)
            h1 = self.b_conv7_1(x)
            h2 = self.b_conv7_2(x)
            return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        if self.architecture is None:
            x = z.view(-1, self.zsize, 1, 1)
            x = F.relu(self.deconv1_bn(self.deconv1(x)))
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = F.relu(self.deconv3_bn(self.deconv3(x)))
            x = F.tanh(self.deconv4(x)) * 0.5 + 0.5
            return x
        
        elif self.architecture == 'b1':
            x = z.view(-1, self.zsize, 1, 1)
            x = F.relu(self.b_deconv1_bn(self.b_deconv1(x)))
            x = F.relu(self.b_deconv2_bn(self.b_deconv2(x)))
            x = F.relu(self.b_deconv3_bn(self.b_deconv3(x)))
            x = F.relu(self.b_deconv4_bn(self.b_deconv4(x)))
            x = F.relu(self.b_deconv5_bn(self.b_deconv5(x)))
            x = F.tanh(self.b_deconv6(x)) * 0.5 + 0.5
            return x

        elif self.architecture == 'b2':
            x = z.view(-1, self.zsize, 1, 1)
            x = F.relu(self.b_deconv1_bn(self.b_deconv1(x)))
            x = F.relu(self.b_deconv2_bn(self.b_deconv2(x)))
            x = F.relu(self.b_deconv3_bn(self.b_deconv3(x)))
            x = F.relu(self.b_deconv4_bn(self.b_deconv4(x)))
            x = F.relu(self.b_deconv5_bn(self.b_deconv5(x)))
            x = F.relu(self.b_deconv6_2_bn(self.b_deconv6_2(x)))
            x = F.tanh(self.b_deconv7(x)) * 0.5 + 0.5  
            return x      

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    # initializers
    def __init__(self, zsize, d=128, channels=1, architecture = None):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(zsize, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)
        
        self.b_deconv1_1 = nn.ConvTranspose2d(zsize, d*8, 4, 1, 0)
        self.b_deconv1_1_bn = nn.BatchNorm2d(d*8)
        self.b_deconv2 = nn.ConvTranspose2d(d*8, d*8, 4, 2, 1)
        self.b_deconv2_bn = nn.BatchNorm2d(d*8)
        self.b_deconv3 = nn.ConvTranspose2d(d*4, d*4, 4, 2, 1)
        self.b_deconv3_bn = nn.BatchNorm2d(d*4)
        self.b_deconv4 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.b_deconv4_bn = nn.BatchNorm2d(d*2)
        self.b_deconv5 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.b_deconv5_bn = nn.BatchNorm2d(d)
        self.b_deconv6_1 = nn.ConvTranspose2d(d, channels, 4, (3,5), (1,0), output_padding=(1,1))
        self.b_deconv_6_2 = self.b_deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.b_deconv_6_2_bn = nn.BatchNorm2d(d//2)
        self.b_deconv7 = nn.ConvTranspose2d(d//2, channels, 4, 2, 1)

        self.architecture = architecture

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):#, label):
        if self.architecture is None:
            x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
            x = F.relu(self.deconv2_bn(self.deconv2(x)))
            x = F.relu(self.deconv3_bn(self.deconv3(x)))
            x = F.tanh(self.deconv4(x)) * 0.5 + 0.5
            return x
        
        elif self.architecture == 'b1':
            x = F.relu(self.b_deconv1_1_bn(self.b_deconv1_1(input)))
            x = F.relu(self.b_deconv2_bn(self.b_deconv2(x)))
            x = F.relu(self.b_deconv3_bn(self.b_deconv3(x)))
            x = F.relu(self.b_deconv4_bn(self.b_deconv4(x)))
            x = F.relu(self.b_deconv5_bn(self.b_deconv5(x)))
            x = F.tanh(self.b_deconv6_1(x)) * 0.5 + 0.5
            return x

        elif self.architectyre == "b2":
            x = F.relu(self.b_deconv1_1_bn(self.b_deconv1_1(input)))
            x = F.relu(self.b_deconv2_bn(self.b_deconv2(x)))
            x = F.relu(self.b_deconv3_bn(self.b_deconv3(x)))
            x = F.relu(self.b_deconv4_bn(self.b_deconv4(x)))
            x = F.relu(self.b_deconv5_bn(self.b_deconv5(x)))
            x = F.relu(self.b_deconv6_2_bn(self.b_deconv6_2(x)))
            x = F.tanh(self.b_deconv7(x)) * 0.5 + 0.5
            return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=1, architecture = None):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4  )
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)
        
        self.b_conv1_1 = nn.Conv2d(channels, d//2, 4, (3,5), 1)
        self.b_conv1_2 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.b_conv2 = nn.Conv2d(d // 2, d, 4, 2, 1)
        self.b_conv2_bn = nn.BatchNorm2d(d)
        self.b_conv3 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.b_conv3_bn = nn.BatchNorm2d(d*2)
        self.b_conv4 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.b_conv4_bn = nn.BatchNorm2d(d*4)
        self.b_conv5 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.b_conv5_bn = nn.BatchNorm2d(d*8)
        self.b_conv6_1 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.b_conv6_2 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.b_conv6_2_bn = nn.BatchNorm2d(d*8)
        self.b_conv7 = nn.Conv2d(d*8, 1, 4, 1, 0)


        self.architecture = architecture
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        if self.architecture is None:
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.sigmoid(self.conv4(x))
            return x

        elif self.architecture == 'b1':
            x = F.leaky_relu(self.b_conv1_1(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.sigmoid(self.b_conv6_1(x))
            return x
        
        elif self.architecture == 'b2':
            x = F.leaky_relu(self.b_conv1_1(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.leaky_relu(self.b_conv6_2_bn(self.b_conv6_2(x)), 0.2)
            x = F.sigmoid(self.b_conv7(x))
            return x


class Encoder(nn.Module):
    # initializers
    def __init__(self, zsize, d=128, channels=1, architecture = None):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, zsize, 4, 1, 0)
        
        self.b_conv1_1 = nn.Conv2d(channels, d//2, 4, (3,5), 1)
        self.b_conv1_2 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.b_conv2 = nn.Conv2d(d // 2, d, 4, 2, 1)
        self.b_conv2_bn = nn.BatchNorm2d(d)
        self.b_conv3 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.b_conv3_bn = nn.BatchNorm2d(d*2)
        self.b_conv4 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.b_conv4_bn = nn.BatchNorm2d(d*4)
        self.b_conv5 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.b_conv5_bn = nn.BatchNorm2d(d*8)
        self.b_conv6_1 = nn.Conv2d(d*8, zsize, 4, 1, 0)
        self.b_conv6_2 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.b_conv6_2_bn = nn.BatchNorm2d(d*8)
        self.b_conv7 = nn.Conv2d(d*8, zsize, 4, 1, 0)

        self.architecture = architecture
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        if self.architecture is None:
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = self.conv4(x)
            return x
        
        elif self.architecture == 'b1':
            x = F.leaky_relu(self.b_conv1_1(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.sigmoid(self.b_conv6_1(x))
            return x

        elif self.architecture == 'b2':
            x = F.leaky_relu(self.b_conv1_2(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.leaky_relu(self.b_conv6_2_bn(self.b_conv6_2(x)), 0.2)
            x = F.sigmoid(self.b_conv7(x))
            return x

class ZDiscriminator(nn.Module):
    # initializers
    def __init__(self, zsize, batchSize, d=128):
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(zsize, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = F.sigmoid(self.linear3(x))
        return x


class ZDiscriminator_mergebatch(nn.Module):
    # initializers
    def __init__(self, zsize, batchSize, d=128):
        super(ZDiscriminator_mergebatch, self).__init__()
        self.linear1 = nn.Linear(zsize, d)
        self.linear2 = nn.Linear(d * batchSize, d)
        self.linear3 = nn.Linear(d, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2).view(1, -1) # after the second layer all samples are concatenated
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = F.sigmoid(self.linear3(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
