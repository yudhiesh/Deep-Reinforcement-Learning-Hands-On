import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # We don't know the exact number of values in the output from the CNN produced with the layer of the given shape.
        # _get_conv_out() takes as input the input shape and applies the CNN operation to a fake tensor of such a shape
        # the result of the function would be equal to the number of parameters returned by this application

        def _get_conv_out(self, shape):
            output = self.conv(torch.zeros(1, *shape))
            return int(np.prod(output.size()))

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        # PyTorch does not have a flatten layer that can transform a 3D vectors into a 1D vectors
        # The problem is solved in the forward()
        # forward() takes as input the 4D input tensor
        # input_shape[0] -> batch_size
        # input_shape[1] -> color channel
        # input_shape[2:] -> image dimensions
        def forward(self, x):
            # First we apply the conv layer to the input
            # Then we obtain the 4D tensor on output
            # The result is then flattened to produce a 2D tensor
            # result[0] -> batch_size
            # result[1] -> parameters returned by the conv for this batch entry
            # This is done by the view()
            # This function which lets one single dimension be a -1 argument as a wildcard for the rest of the parameters
            # Finally we pass in this flattened 2D tensor to the fc layer to obtain the Q-values for every batch input
            conv_out = self.conv(x).view(x.size()[0], -1)
            return self.fc(conv_out)
