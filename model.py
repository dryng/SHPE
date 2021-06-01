import torch
import torch.nn as nn
import torch.nn.functional as F


'''
return a resnet model, initialized with Xavier uniform weight initialization, of either size 18 or 34
params:
    int size: size of model - 18 or 34
returns:
    model 
'''
def get_model_resnet(size, device):
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    if size == 18:
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
    elif size == 34:
        b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 4))
        b4 = nn.Sequential(*resnet_block(128, 256, 6))
        b5 = nn.Sequential(*resnet_block(256, 512, 3))
    else:
        raise ValueError('Incorrect model size! Available models: 18 or 34 layers')

    model_wo_fc = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten())

    model = MultiLossModel(model_wo_fc).to(device)

    # 30 -> 15 x,y coordinates

    # model.apply(initialize_weights)
    model.apply(initialize_weights_kaming)

    return model


def get_model_resnet_no_vis(size, device):
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    if size == 18:
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
    elif size == 34:
        b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 4))
        b4 = nn.Sequential(*resnet_block(128, 256, 6))
        b5 = nn.Sequential(*resnet_block(256, 512, 3))
    else:
        raise ValueError('Incorrect model size! Available models: 18 or 34 layers')

    model = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten(), nn.Linear(512, 32)).to(device)

    # 30 -> 15 x,y coordinates

    # model.apply(initialize_weights)
    model.apply(initialize_weights_kaming)

    return model


class MultiLossModel(nn.Module):
    def __init__(self, model_wo_fc):
        super().__init__()
        self.model_wo_fc = model_wo_fc
        self.fc = nn.Linear(512, 48)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.model_wo_fc(x)
        x = self.fc(x)
        x = torch.cat((self.sigmoid(x[:,0:16]), x[:,16:]), dim=1)
        return x

'''
The residual block has two 3×3 convolutional layers with the same number of output channels.
Each convolutional layer is followed by a batch normalization layer and a ReLU activation function.
Then, skip these two convolution operations and add the input directly before the final ReLU activation function.
This kind of design requires that the output of the two convolutional layers has to be of the same shape
as the input, so that they can be added together. To change number of channels,
you need to introduce an additional 1×1 convolutional layer to transform the input into the
desired shape for the addition operation.
'''
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        # shape = (N + 2 - 3)/Stride + 1
        # i.e 64x64 & stride = 1 -> (64 + 2 - 3)/1 + 1 = 64x64
        # i.e 64x64 & stride = 2 -> (64 + 2 - 3)/2 + 1 = 32.5x32.5? -> next block has same
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)  # why no stride = keep size same
        # i.e 64x64 & stride = 1 -> (64 + 2 - 3)/1 + 1 = 64x64
        # shape = (N + 2 - 3)/1 + 1
        if use_1x1:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            # i.e 64x64 & stride = 1 -> (64 - 1)/1 + 1 = 64x64
            # shape = (N + 2*(0) - 1)/Stride + 1
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # x = 64x64x3
        # i.e 64x64, input = 3, ouput = 6 ->
        y = F.relu(self.bn1(self.conv1(x)))  # 64x64x6
        y = self.bn2(self.conv2(y))  # 64x64x6
        # can't add 64x64x3 and 64x64x6
        if self.conv3:  # if num_channels != input_channels
            # this transforms input to output size!!!
            x = self.conv3(x)
        y += x  # add og back        #64x64x6 + 64x64x6
        return F.relu(y)


'''
ResNet uses four modules made up of residual blocks, 
each of which uses several residual blocks with the same number of output channels. 
The number of channels in the first module is the same as the number of input channels. 
Since a maximum pooling layer with a stride of 2 has already been used, 
it is not necessary to reduce the height and width.

For the first residual within the block, the number of channels is doubled and the height and width are halved.
'''
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1=True,
                               strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


'''
Intialize model with Xavier uniform weight initialization
Params:
    model
'''
def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data, gain=nn.init.calculate_gain('relu'))

'''
Intialize model with values according to the method described in Delving deep into rectifiers
Params:
    model
'''
def initialize_weights_kaming(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')

'''
Return the amount of parameters the given model has
Params:
    model
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
Observe the shape of an input image across layers
Params:
    model
    image_size
    device: need device to put fake input image tensor on gpu with model 
'''
def observe_shape(model, image_size, device):
    x = torch.randn(1,3,image_size,image_size).to(device)
    print(f"Input: {x.shape}")
    for layer in model:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:\t', x.shape)