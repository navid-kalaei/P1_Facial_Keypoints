## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # an architect inspired by NaimishNet has been developed
        # four conv stage folowing by maxpooling and dropouts
        # as the papare suggests the dropout propabilities are between 0.1 and 0.6 with a step size of 0.1
        # how ever to avoid complexity I reduced the number of conv layers from 6 to 4 and began the dropout prob from 0.3
        # https://arxiv.org/pdf/1710.00977.pdf
        
        # each pooling dived the size by 2
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # from the last expriences it is best to increase the number of channels by a factor of 2 and as the size of features reduces by pooling it is better to reduces size of the window of conving layer too
        
        #conv layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout2d(0.3)
        
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.drop2 = nn.Dropout2d(0.3)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout2d(0.4)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.drop4 = nn.Dropout2d(0.4)
        
        # fully connected layers
        self.fc1 = nn.Linear(11*11*256, 1000)
        self.drop5 = nn.Dropout2d(0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout2d(0.6)
        
        # last layer with the maximum number of key points
        self.fc3 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
