import torch.nn as nn

def crossEntropyLoss():
    return nn.CrossEntropyLoss.cuda()