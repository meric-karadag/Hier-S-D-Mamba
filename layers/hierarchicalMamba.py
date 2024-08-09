from mamba_ssm import Mamba
import torch
from torch import nn
import torch.functional as F
from hierarchicalMambaBlocks import MambaMain, MambaWithExternalDt

class HierarchicalMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.model1 = MambaMain(d_model, **kwargs)
        self.model2 = MambaWithExternalDt(d_model, **kwargs)
        self.model3 = MambaWithExternalDt(d_model, **kwargs)
        self.model4 = MambaWithExternalDt(d_model, **kwargs)

    def forward(self, hidden_states):
        is_2d = len(hidden_states.shape) == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(-1)
    
        output1, dt, dt_bias = self.model1(hidden_states)

        dt_2 = dt * 2
        dt_bias_2 = dt_bias * 2
        dt_4 = dt * 4
        dt_bias_4 = dt_bias * 4
        dt_8 = dt * 8
        dt_bias_8 = dt_bias * 8
    
        output2 = self.model2(hidden_states, dt_2, dt_bias_2)
        output3 = self.model3(hidden_states, dt_4, dt_bias_4)
        output4 = self.model4(hidden_states, dt_8, dt_bias_8)

        output = torch.mean(torch.stack([output1, output2, output3, output4]), dim=0, keepdim=False)
        return output
