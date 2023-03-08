import torch

class ConvBlock(torch.nn.Module):
    blockparents = []
    name = ""
    def __init__(self, modules, name):
        super(ConvBlock, self).__init__()
        self.blockmodules = torch.nn.Sequential(*modules)
        self.name = name

    def forward(self, x):
        return self.blockmodules(x)

    def setParents(self, parents):
        assert isinstance(parents, list)
        self.blockparents = parents

