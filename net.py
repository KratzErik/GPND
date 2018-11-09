
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
    def __init__(self, zsize, d=128, db=8, channels=1, architecture = None, h_out):
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
        
        self.b_deconv1_1 = nn.ConvTranspose2d(zsize, db*32, 4, 1, 0)
        self.b_deconv1_1_bn = nn.BatchNorm2d(db*32)
        self.b_deconv2 = nn.ConvTranspose2d(db*32, db*16, 4, 2, 1)
        self.b_deconv2_bn = nn.BatchNorm2d(d*16)
        self.b_deconv3 = nn.ConvTranspose2d(d*16, d*8, 4, 2, 1)
        self.b_deconv3_bn = nn.BatchNorm2d(d*8)
        self.b_deconv4 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.b_deconv4_bn = nn.BatchNorm2d(d*4)
        self.b_deconv5 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.b_deconv5_bn = nn.BatchNorm2d(d*2)
        self.b_deconv6_1 = nn.ConvTranspose2d(d, channels, 4, (3,5), (1,0), output_padding=(1,1))
        self.b_deconv_6_2 = self.b_deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.b_deconv_6_2_bn = nn.BatchNorm2d(d//2)
        self.b_deconv7 = nn.ConvTranspose2d(d//2, channels, 4, 2, 1)

        self.architecture = architecture

        # This allows for passing a string that sets the network architecture automatically
        # 'x_A_B_C_D', e.g. s_6_0_16_512
        #x	's' or 'p' if dim-reduction is done with 2x2 stride ('s') or 2x2 max pooling ('p')	
        #A	int	number of conv. layers (in either encoder/decoder)	
        #B	int	number of dense layers, should be 0 or 1
        #C	int	Channels out from first conv. layer (will double every layer)	
        #D	int	Latent dim

        if self.architecture not in [None,"b1","b2"]:
            # Parse configuration string
            tmp = self.architecture.split("_")
            use_pool = tmp[0]
            n_deconv = int(tmp[1])
            n_dense = int(tmp[2]) # B: should be 0 or 1 
            db = int(tmp[3])
            zsize = int(tmp[4])

            # Add dense layer, with correct input and output dimensions
            if n_dense > 0:
                dim_out_dense = h_out**2 * db //(2**(n_deconv+1))
                self.dense_layer = nn.Linear(zsize,dim_out_dense)
                self.dense_bn = nn.BatchNorm1D(dim_out_dense)      
            else:
                self.dense_layer = None
            
            # Add deconvolutional layers, all but the last with batch norm
            self.deconv_layers = []
            self.bn_layers = []
            c_in = db * (2**(n_deconv-1))
            h_in = h_out/(2**n_deconv)

            for i in range(n_deconv-1):
                if i == 0: # First deconv layer maps 1x1 to 4x4 image
                    self.deconv_layers.append(nn.ConvTranspose2d(c_in, c_in//2, 4, 1, 0))
                else:
                    self.deconv_layers.append(nn.ConvTranspose2d(c_in, c_in//2, 4, 2, 1))
                
                self.bn_layers.append(nn.BatchNorm2D(c_in//2))
                c_in //= 2
            
            # Final conv layer
            self.output_layer = nn.ConvTranspose2d(c_in, channels, 4, 2, 1)

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
        
        # hard coded architectures
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

        else: # build architecture from spec: x_A_B_C_D
            # height of image at start of deconvolutions
            if self.dense_layer is not None:
                x = F.relu(self.dense_bn(self.dense_layer(input)))
            else: 
                x = input
            
            for bn_layer, deconv_layer in zip(self.bn_layers,self.deconv_layers):
                x = F.relu(bn_layer(deconv_layer(x)))

            x = F.tanh(self.output_layer(x))*0.5 + 0.5

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, db=8, channels=1, architecture = None, h_in):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4  )
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)
        
        self.b_conv1_1 = nn.Conv2d(channels, db, 4, (3,5), 1)
        self.b_conv1_2 = nn.Conv2d(channels, db, 4, 2, 1)
        self.b_conv2 = nn.Conv2d(db, db*2, 4, 2, 1)
        self.b_conv2_bn = nn.BatchNorm2d(db*2)
        self.b_conv3 = nn.Conv2d(db*2, db*4, 4, 2, 1)
        self.b_conv3_bn = nn.BatchNorm2d(db*4)
        self.b_conv4 = nn.Conv2d(db*4, db*8, 4, 2, 1)
        self.b_conv4_bn = nn.BatchNorm2d(db*8)
        self.b_conv5 = nn.Conv2d(db*8, db*16, 4, 2, 1)
        self.b_conv5_bn = nn.BatchNorm2d(db*16)
        self.b_conv6_1 = nn.Conv2d(db*16, 1, 4, 1, 0)
        self.b_conv6_2 = nn.Conv2d(db*16, db*32, 4, 2, 1)
        self.b_conv6_2_bn = nn.BatchNorm2d(db*32)
        self.b_conv7 = nn.Conv2d(db*32, 1, 4, 1, 0)


        self.architecture = architecture

        if self.architecture not in [None,"b1","b2"]:
            tmp = self.architecture.split("_")
            dim_red_type = tmp[0]
            n_conv = int(tmp[1])
            n_dense = int(tmp[2]) # B: should be 0 or 1 
            c_out = int(tmp[3])

            self.input_layer = nn.Conv2d(channels, c_out, 4, 2, 1)
            self.input_bn = nn.BatchNorm2D(c_out)
            
            self.conv_layers = []
            self.bn_layers = []

            for i in range(n_conv-2):
                c_out *= 2
                self.conv_layers.append(nn.Conv2d(c_out//2, c_out, 4, 2, 1))
                self.bn_layers.append(nn.BatchNorm2D(c_out))
            
            # Final conv layer without batch norm
            c_out *= 2
            self.encoding_layer = nn.Conv2d(c_out//2, c_out, 4, 2, 1)

            current_h = h_in//(2**(n_conv))
            if current_h > 1:
                if n_dense == 1:
                    # Output layer is dense
                    self.encoding_bn = nn.BatchNorm2D(c_out)
                    tot_dim = current_h**2 * c_out
                    self.dense_layer = nn.Linear(tot_dim,1)
                elif current_h == 2:
                    # Maps 4x4 image to 1x1 instead of 2x2 (current)
                    self.self.encoding_layer = nn.Conv2d(c_out//2, 1, 4, 1, 0)
                    self.dense_layer = None
                else:
                    print("Number of conv. layers wrong. If # dense layers is 0 the number of conv. layers should result in an output image of h=w=1.")
                    exit()                

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

        else:
            x = F.leaky_relu(self.input_bn(self.input_layer(input)))
            
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = bn(conv(x))
            x = self.encoding_layer(x)
            if self.dense_layer is not None:
                x = self.encoding_bn(x)
                x = x.view(x.numel())
                x = self.dense_layer(x)

            return F.sigmoid(x)


class Encoder(nn.Module):
    # initializers
    def __init__(self, zsize, d=128, db = 8, channels=1, architecture = None):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, zsize, 4, 1, 0)
        
        self.b_conv1_1 = nn.Conv2d(channels, db, 4, (3,5), 1)
        self.b_conv1_2 = nn.Conv2d(channels, db, 4, 2, 1)
        self.b_conv2 = nn.Conv2d(db, db*2, 4, 2, 1)
        self.b_conv2_bn = nn.BatchNorm2d(db*2)
        self.b_conv3 = nn.Conv2d(db*2, db*4, 4, 2, 1)
        self.b_conv3_bn = nn.BatchNorm2d(db*4)
        self.b_conv4 = nn.Conv2d(db*4, db*8, 4, 2, 1)
        self.b_conv4_bn = nn.BatchNorm2d(db*8)
        self.b_conv5 = nn.Conv2d(db*8, db*16, 4, 2, 1)
        self.b_conv5_bn = nn.BatchNorm2d(db*16)
        self.b_conv6_1 = nn.Conv2d(db*16, zsize, 4, 1, 0)
        self.b_conv6_2 = nn.Conv2d(db*16, db*32, 4, 2, 1)
        self.b_conv6_2_bn = nn.BatchNorm2d(d*32)
        self.b_conv7 = nn.Conv2d(d*32, zsize, 4, 1, 0)

        self.architecture = architecture

        if self.architecture not in [None,"b1","b2"]:
            tmp = self.architecture.split("_")
            dim_red_type = tmp[0]
            n_conv = int(tmp[1])
            n_dense = int(tmp[2]) # B: should be 0 or 1 
            c_out = int(tmp[3])
            zsize = int(tmp[4])

            self.input_layer = nn.Conv2d(channels, c_out, 4, 2, 1)
            self.input_bn = nn.BatchNorm2D(c_out)
            
            self.conv_layers = []
            self.bn_layers = []

            for i in range(n_conv-1):
                c_out *= 2
                self.conv_layers.append(nn.Conv2d(c_out//2, c_out, 4, 2, 1))
                

            current_h = h_in//(2**(n_conv))
            if current_h > 1:
                if n_dense == 1:
                    self.bn_layers.append(nn.BatchNorm2D(c_out))
                    # Output layer is dense
                    tot_dim = current_h**2 * c_out
                    self.dense_layer = nn.Linear(tot_dim,zsize)
                elif current_h == 2:
                    # Maps 4x4 image to 1x1 instead of 2x2 (current)
                    self.conv_layers[-1] = nn.Conv2d(c_out//2, zsize, 4, 1, 0)
                    self.dense_layer = None
                else:
                    print("Number of conv. layers wrong. If # dense layers is 0 the number of conv. layers should result in an output image of h=w=1.")
                    exit()   
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
            x = self.b_conv6_1(x)
            return x

        elif self.architecture == 'b2':
            x = F.leaky_relu(self.b_conv1_2(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.leaky_relu(self.b_conv6_2_bn(self.b_conv6_2(x)), 0.2)
            x = self.b_conv7(x)
            return x

        else:
            x = F.leaky_relu(self.input_bn(self.input_layer(input)))
            
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = bn(conv(x))
            
            if self.dense_layer is not None:
                x = self.dense_layer(x)

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
