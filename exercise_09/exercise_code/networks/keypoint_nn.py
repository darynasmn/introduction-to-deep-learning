"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # You're going probably try different architectures, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        self.conv1 = nn.Conv2d( hparams["in_channels"], hparams["conv_channels"][0], kernel_size=hparams["kernel_size"], padding=hparams["padding"])
        self.batch1 = nn.BatchNorm2d(hparams["conv_channels"][0])

        self.conv2 = nn.Conv2d(hparams["conv_channels"][0], hparams["conv_channels"][1], kernel_size=hparams["kernel_size"], padding=hparams["padding"])
        self.batch2 = nn.BatchNorm2d(hparams["conv_channels"][1])

        self.conv3 = nn.Conv2d(hparams["conv_channels"][1], hparams["conv_channels"][2], kernel_size=hparams["kernel_size"], padding=hparams["padding"])
        self.batch3 = nn.BatchNorm2d(hparams["conv_channels"][2])
        self.dropout3 = nn.Dropout2d(p=hparams["dropout_conv"])

        self.conv4 = nn.Conv2d(hparams["conv_channels"][2], hparams["conv_channels"][3], kernel_size=hparams["kernel_size"], padding=hparams["padding"])
        self.batch4 = nn.BatchNorm2d(hparams["conv_channels"][3])
        self.dropout4 = nn.Dropout2d(p=hparams["dropout_conv"])

        self.conv5 = nn.Conv2d(hparams["conv_channels"][3], hparams["conv_channels"][4], kernel_size=hparams["kernel_size"], padding=hparams["padding"])
        self.batch5 = nn.BatchNorm2d(hparams["conv_channels"][4])
        self.dropout5 = nn.Dropout2d(p=hparams["dropout_conv"])

        final_spatial = hparams["input_size"] // (2 ** 5)

        self.fc1 = nn.Linear(
            hparams["conv_channels"][-1] * final_spatial * final_spatial,
            hparams["fc_units"]
        )
        self.dropout_L = nn.Dropout(p=hparams["dropout_fc"])

        self.fc2 = nn.Linear(
            hparams["fc_units"],
            hparams["num_keypoints"] * 2
        )

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

        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.hparams["pool_kernel"])

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.hparams["pool_kernel"])

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = F.max_pool2d(x, self.hparams["pool_kernel"])

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = F.max_pool2d(x, self.hparams["pool_kernel"])

        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = F.max_pool2d(x, self.hparams["pool_kernel"])

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_L(x)
        x = self.fc2(x)          

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
