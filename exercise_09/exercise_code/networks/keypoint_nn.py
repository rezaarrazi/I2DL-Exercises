"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        # Define the layers
        self.conv1 = nn.Conv2d(1, 16, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(hparams['dropout'])
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(hparams['dropout'])
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(hparams['dropout'])
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(hparams['dropout'])
        self.act4 = nn.ReLU()
        
        self.fc1 = nn.Linear(128*5*5, 500)
        self.drop5 = nn.Dropout(hparams['dropout'])
        self.act5 = nn.ReLU()
        
        self.fc2 = nn.Linear(500, 500)
        self.drop6 = nn.Dropout(hparams['dropout'])
        self.act6 = nn.ReLU()
        
        self.fc3 = nn.Linear(500, 30)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################


        x = self.drop1(self.pool1(self.act1(self.conv1(x))))
        x = self.drop2(self.pool2(self.act2(self.conv2(x))))
        x = self.drop3(self.pool3(self.act3(self.conv3(x))))
        x = self.drop4(self.pool4(self.act4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.drop5(self.act5(self.fc1(x)))
        x = self.drop6(self.act6(self.fc2(x)))
        x = self.fc3(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
