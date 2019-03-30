import torch
import torch.nn as nn
import torch.nn.functional as F

class OHLMlp(nn.Module):
    def __init__(self):
        super (OHLMlp,self).__init__()
        self.Hidden = nn.Linear(28*28,500)
        self.fc2=nn.Linear(500,10)

    def forward(self, x):
        x=torch.flatten(x,start_dim=1)
        x=self.Hidden(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x