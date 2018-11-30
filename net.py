import torch
from torch import nn
from torch.nn import functional as F
from configuration import Configuration as cfg

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

        if self.architecture not in [None,"b1","b2"]:
            # architecture spec A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            if use_pool:
                pad = 0
            print("Constructing VAE")
            self.conv_layers = []
            self.encoding_bn_layers = []
            self.pool_layers = []

            self.input_layer = nn.Conv2d(channels, num_filters, ksize, stride, pad)
            self.input_bn = nn.BatchNorm2d(num_filters)
            self.input_pool = nn.MaxPool2d((2,2))
            print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            for i in range(n_conv-2):
                num_filters *= 2
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.encoding_bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))

                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            # Final conv layer without batch norm
            num_filters *= 2
            if n_dense > 0:
                # Add final conv_layer
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.encoding_bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))
                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

                # Add dense layer
                num_dense_in =  c_1 * (2**(n_conv-1))
                self.encoding_dense_layer_1 = nn.Linear(num_filters,zsize)
                self.encoding_dense_layer_2 = nn.Linear(num_filters,zsize)
                print("\tAdded encoding dense layer")
            else:
                # Add final conv_layer:
                h = self.image_height // (2**(n_conv-1))
                self.encoding_convlayer_1 = nn.Conv2d(num_filters//2, zsize, h, 1, 0)
                self.encoding_convlayer_2 = nn.Conv2d(num_filters//2, zsize, h, 1, 0)
                print("\tAdded conv_layer %d (encoding)" % (len(self.conv_layers)+2))

            # Decoding layers
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_out = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_out
            outpad = 0

            if use_pool:
                # Compute output padding and padding so that output size is same as input for deconv-layers
                if stride != 1:
                    print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
                else:
                    outpad = (ksize-stride)%2
                    pad = (ksize-stride+outpad)//2
            else: # using stride to increase image size: set pad and outpad so that h_out = stride * h_in
                if stride == 1:
                    print("No pool and stride = 1. Image size will not increase")
                else:
                   outpad = (ksize-stride)%2
                   pad = (ksize-stride+outpad)//2

            self.deconv_layers = []
            self.decoding_bn_layers = []
            if n_dense > 0:

                h1 = self.image_height // (2**n_conv) # height = width of image going into first conv layer
                num_filters =  c_out * (2**(n_conv-1))
                num_out_dense_units = h1**2 * num_filters
                self.decoding_dense_layer = nn.Linear(zsize,num_out_dense_units)
                self.decoding_dense_bn = nn.BatchNorm1d(num_out_dense_units)
                print("\tAdded dense layer")

                num_filters = num_filters // 2

                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))
                
                self.deconv_layers.append(nn.ConvTranspose2d(num_filters*2, num_filters, ksize, stride, pad, output_padding = outpad ))
                self.decoding_bn_layers.append(nn.BatchNorm2d(num_filters))

                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))

                num_filters //=2

            else:
                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))

                h2 = self.image_height // (2**(n_conv-1)) # height of image going in to second conv layer
                num_filters = c_out * (2**(n_conv-2))
                self.deconv_layers.append(nn.ConvTranspose2d(num_filters*2, num_filters, h2, 1, 0))
                self.decoding_bn_layers.append(nn.BatchNorm2d(num_filters))
                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))

            # Add remaining deconv layers
            for i in range(n_conv-2):
                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))

                self.deconv_layers.append(nn.ConvTranspose2d(num_filters*2, num_filters, ksize, stride, pad, output_padding = outpad ))
                self.decoding_bn_layers.append(nn.BatchNorm2d(num_filters))

                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))
                num_filters //=2

            # add reconstruction layer
            #if use_pool:
            #    self.final_upsample = F.interpolate(scale_factor = 2, mode = 'nearest')

            self.output_layer = nn.ConvTranspose2d(num_filters*2, self.channels, ksize, stride, pad)
            print("\tAdded deconv_layer %d (reconstruction)" % (len(self.deconv_layers)+1))

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

        else:
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            x = self.input_layer(input)

            if cfg.use_batchnorm:
                x = self.input_bn(x)

            x = F.leaky_relu(x)

            for conv, bn, pool in zip(self.conv_layers, self.encoding_bn_layers, self.pool_layers):
                x = conv(x)
                if cfg.use_batchnorm:
                    x = bn(x)
                if use_pool:
                    x = pool(x)

            if n_dense > 0:
                x = x.view(x.numel())
                h1 = self.encoding_dense_layer_1(x)
                h2 = self.encoding_dense_layer_2(x)
            else:
                h1 = self.encoding_convlayer_1(x)
                h2 = self.encoding_convlayer_2(x)

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

        else: # build architecture from spec: A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_out = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_out

            if n_dense > 0:
                h1 = self.image_height // (2**n_conv) # height = width of image going into first conv layer
                num_filters =  c_out * (2**(n_conv-1))
                x = F.relu(self.decoding_dense_bn(self.decoding_dense_layer(input)))
                x = x.view(-1,num_filters,h1,h1)
            else:
                x = input


            for bn, deconv in zip(self.decoding_bn_layers,self.deconv_layers):
                if use_pool:
                    x = F.interpolate(x,scale_factor = 2, mode = 'nearest')
                x = F.relu(bn(deconv(x)))

            if use_pool:
                x = F.interpolate(x, scale_factor = 2, mode = 'nearest')

            x = F.tanh(self.output_layer(x))*0.5 + 0.5

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
    def __init__(self, zsize, d=128, db=8, channels=1, architecture = None):
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

        self.b_deconv1_1     = nn.ConvTranspose2d(zsize, db*32, 4, 1, 0)
        self.b_deconv1_1_bn  = nn.BatchNorm2d(db*32)
        self.b_deconv2       = nn.ConvTranspose2d(db*32, db*16, 4, 2, 1)
        self.b_deconv2_bn    = nn.BatchNorm2d(db*16)
        self.b_deconv3       = nn.ConvTranspose2d(db*16, db*8, 4, 2, 1)
        self.b_deconv3_bn    = nn.BatchNorm2d(db*8)
        self.b_deconv4       = nn.ConvTranspose2d(db*8, db*4, 4, 2, 1)
        self.b_deconv4_bn    = nn.BatchNorm2d(db*4)
        self.b_deconv5       = nn.ConvTranspose2d(db*4, db*2, 4, 2, 1)
        self.b_deconv5_bn    = nn.BatchNorm2d(db*2)
        self.b_deconv6_1     = nn.ConvTranspose2d(db*2, channels, 4, (3,5), (1,0), output_padding=(1,1))
        self.b_deconv6_2    = nn.ConvTranspose2d(db*2, db, 4, 2, 1)
        self.b_deconv6_2_bn = nn.BatchNorm2d(db)
        self.b_deconv7       = nn.ConvTranspose2d(db, channels, 4, 2, 1)

        self.architecture = architecture

        # This allows for passing a string that sets the network architecture automatically
        # 'x_A_B_C_D', e.g. s_6_0_16_512
        #x	's' or 'p' if dim-reduction is done with 2x2 stride ('s') or 2x2 max pooling ('p')
        #A	int	number of conv. layers (in either encoder/decode)
        #B	int	number of dense layers, should be 0 or 1
        #C	int	Channels out from first conv. layer (will double every layer)
        #D	int	Latent dim

        if self.architecture not in [None,"b1","b2"]:
            print("Constructing generator G()")
            # architecture spec A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_out = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_out
            outpad = 0

            if use_pool:
                # Compute output padding and padding so that output size is same as input for deconv-layers
                if stride != 1:
                    print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
                else:
                    outpad = (ksize-stride)%2
                pad = (ksize-stride+outpad)//2
            else: # using stride to increase image size: set pad and outpad so that h_out = stride * h_in
                if stride == 1:
                    print("No pool and stride = 1. Image size will not increase")
                else:
                   outpad = (ksize-stride)%2
                   pad = (ksize-stride+outpad)//2

            self.deconv_layers = []
            self.bn_layers = []
            if n_dense > 0:

                h1 = cfg.image_height // (2**n_conv) # height = width of image going into first conv layer
                num_filters =  c_out * (2**(n_conv-1))
                num_out_dense_units = (h1**2) * num_filters
                self.dense_layer = nn.Linear(zsize,num_out_dense_units)
                self.dense_bn = nn.BatchNorm1d(num_out_dense_units)
                print("\tAdded dense layer")
                num_filters = num_filters // 2

                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))

                self.deconv_layers.append(nn.ConvTranspose2d(num_filters*2, num_filters, ksize, stride, pad, output_padding = outpad ))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))

                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))

                num_filters //=2

            else:
                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))

                h2 = cfg.image_height // (2**(n_conv-1)) # height of image going in to second conv layer
                num_filters = c_out * (2**(n_conv-2))
                self.deconv_layers.append(nn.ConvTranspose2d(zsize, num_filters, h2, 1, 0))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))
                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))
                num_filters //=2

            # Add remaining deconv layers
            for i in range(n_conv-2):
                #if use_pool:
                #    self.upsample_layers = []
                #    self.upsample_layers.append(F.interpolate(scale_factor = 2, mode = 'nearest'))

                self.deconv_layers.append(nn.ConvTranspose2d(num_filters*2, num_filters, ksize, stride, pad, output_padding = outpad ))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))

                print("\tAdded deconv_layer %d" % (len(self.deconv_layers)))
                num_filters //=2

            # add reconstruction layer
            #if use_pool:
            #    self.final_upsample = F.interpolate(scale_factor = 2, mode = 'nearest')

            self.output_layer = nn.ConvTranspose2d(num_filters*2, cfg.channels, ksize, stride, pad, output_padding = outpad)
            print("\tAdded deconv_layer %d (reconstruction)" % (len(self.deconv_layers)+1))

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

        elif self.architecture == "b2":
            x = F.relu(self.b_deconv1_1_bn(self.b_deconv1_1(input)))
            x = F.relu(self.b_deconv2_bn(self.b_deconv2(x)))
            x = F.relu(self.b_deconv3_bn(self.b_deconv3(x)))
            x = F.relu(self.b_deconv4_bn(self.b_deconv4(x)))
            x = F.relu(self.b_deconv5_bn(self.b_deconv5(x)))
            x = F.relu(self.b_deconv6_2_bn(self.b_deconv6_2(x)))
            x = F.tanh(self.b_deconv7(x)) * 0.5 + 0.5
            return x

        else: # build architecture from spec: A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_out = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_out

            # height of image at start of deconvolutions
            if n_dense > 0:
                input = input.permute(0,2,3,1)
                h1 = cfg.image_height // (2**n_conv) # height = width of image going into first conv layer
                num_filters =  c_out * (2**(n_conv-1))
                x = self.dense_layer(input)
                x = x.permute(0,3,1,2)
                x = x.view(-1,h1**2 * num_filters)
                x = self.dense_bn(x)
                x = F.relu(x)
                x = x.view(-1,num_filters,h1,h1)
            else:
                x = input
            for bn, deconv in zip(self.bn_layers,self.deconv_layers):
                if use_pool:
                    x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
                x = F.relu(bn(deconv(x)))
            if use_pool:
                x = F.interpolate(x, scale_factor = 2, mode = 'nearest')

            x = F.tanh(self.output_layer(x))*0.5 + 0.5
            return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, db=8, channels=1, architecture = None):
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
            print("Constructing discriminator D()")
            # architecture spec A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            if use_pool:
                pad = 0

            self.conv_layers = []
            self.bn_layers = []
            self.pool_layers = []

            self.input_layer = nn.Conv2d(channels, num_filters, ksize, stride, pad)
            self.input_bn = nn.BatchNorm2d(num_filters)
            self.input_pool = nn.MaxPool2d((2,2))
            print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            for i in range(n_conv-2):
                num_filters *= 2
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))

                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            # Final conv layer without batch norm
            num_filters *= 2
            if n_dense > 0:
                # Add final conv_layer
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))
                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

                # Add dense layer
                num_dense_in =  c_1 * (2**(n_conv-1))
                self.dense_layer = nn.Linear(num_filters,1)
                print("\tAdded output dense layer")
            else:
                # Add final conv_layer:
                h = cfg.image_height // (2**(n_conv-1))
                self.output_convlayer = nn.Conv2d(num_filters//2, 1, h, 1, 0)
                print("\tAdded conv_layer %d (output)" % (len(self.conv_layers)+2))

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
            x = F.leaky_relu(self.b_conv1_2(input), 0.2)
            x = F.leaky_relu(self.b_conv2_bn(self.b_conv2(x)), 0.2)
            x = F.leaky_relu(self.b_conv3_bn(self.b_conv3(x)), 0.2)
            x = F.leaky_relu(self.b_conv4_bn(self.b_conv4(x)), 0.2)
            x = F.leaky_relu(self.b_conv5_bn(self.b_conv5(x)), 0.2)
            x = F.leaky_relu(self.b_conv6_2_bn(self.b_conv6_2(x)), 0.2)
            x = F.sigmoid(self.b_conv7(x))
            return x

        else:
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            x = self.input_layer(input)

            if cfg.use_batchnorm:
                x = self.input_bn(x)

            x = F.leaky_relu(x)

            for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
                x = conv(x)
                if cfg.use_batchnorm:
                    x = bn(x)
                if use_pool:
                    x = pool(x)

            if n_dense > 0:
                x = x.view(x.numel())
                x = self.dense_layer(x)
            else:
                x = self.output_convlayer(x)

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
        self.b_conv6_2_bn = nn.BatchNorm2d(db*32)
        self.b_conv7 = nn.Conv2d(db*32, zsize, 4, 1, 0)

        self.architecture = architecture

        if self.architecture not in [None,"b1","b2"]:
            print("Constructing encoder E()")
            # architecture spec A_B_C_D_E_F_G_H
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            if use_pool:
                if stride != 1:
                    print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
                else: # stride is 1
                    pad = (ksize-1)//2
            else:
                if stride == 1:
                    print("Warning: no pooling and stride = 1. Image size will not change.")
                else: # assuming dilation = 1, ksize uneven, image height = width divisible by ksize
                    pad = (ksize-stride)//2+(ksize-stride)%2

            self.conv_layers = []
            self.bn_layers = []
            self.pool_layers = []

            self.input_layer = nn.Conv2d(channels, num_filters, ksize, stride, pad)
            self.input_bn = nn.BatchNorm2d(num_filters)
            self.input_pool = nn.MaxPool2d((2,2))
            print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            for i in range(n_conv-2):
                num_filters *= 2
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))

                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

            # Final conv layer without batch norm
            num_filters *= 2
            if n_dense > 0:
                # Add final conv_layer
                self.conv_layers.append(nn.Conv2d(num_filters//2, num_filters, ksize, stride, pad))
                self.bn_layers.append(nn.BatchNorm2d(num_filters))
                self.pool_layers.append(nn.MaxPool2d((2,2)))
                print("\tAdded conv_layer %d" % (len(self.conv_layers)+1))

                # Add dense layer
                self.num_dense_in =  c_1 * cfg.image_height**2 // (2**(n_conv+1))
                self.dense_layer = nn.Linear(self.num_dense_in,zsize)
                print("\tAdded encoding dense layer")

            else:
                # Add final conv_layer:
                h = cfg.image_height // (2**(n_conv-1))
                self.output_convlayer = nn.Conv2d(num_filters//2, zsize, h, 1, 0)
                print("\tAdded conv_layer %d (encoding)" % (len(self.conv_layers)+2))

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
            tmp = cfg.architecture.split("_")
            use_pool = int(tmp[0]) == 1 # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_1

            x = self.input_layer(input)

#            print("input: ",input.shape)
#            print("x: ",x.shape)

            if cfg.use_batchnorm:
                x = self.input_bn(x)

            if use_pool:
                x = self.input_pool(x)

            x = F.leaky_relu(x)

            for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
                x = conv(x)
#                print("x: ",x.shape)
                if cfg.use_batchnorm:
                    x = bn(x)
                if use_pool:
                    x = pool(x)

            if n_dense > 0:
                x = x.view(-1, 1,1,self.num_dense_in)

#                print("into dense: ", x.shape)
                x = self.dense_layer(x).permute(0,3,1,2)
            else:
                x = self.output_convlayer(x)
#            print("Output: ", x.shape)
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
