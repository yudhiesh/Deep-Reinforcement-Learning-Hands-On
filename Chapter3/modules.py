import torch
import torch.nn as nn


class OurModule(nn.Module):
    # This is the module class that inherits the nn.Module
    # In the constructor we pass in three parameters
    # Then we call the parents constructor to let it intialize itself
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1),
        )

    # Here we override the forward function with our own implementation of data transformation
    # To apply the module to the data we need to call the module as callable
    # DO NOT USE THE forward() of the nn.Module class
    # This is due to the nn.Module overriding the __call__() method
    # If we were to call forward() directly then we would intervene with the nn.Module which can give us wrong results
    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(net)
    print(out)
