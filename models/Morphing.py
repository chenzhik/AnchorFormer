import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class AdaptiveInstanceNorm1d(nn.Module):
    """
    input:
    - inp: (b, c, m)
    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class GridDecoder(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)
    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 2,
        bottleneck_size: int = 1026, 
    ):
        super(GridDecoder, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim

        self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.th = nn.Tanh()

        self.adain1 = AdaptiveInstanceNorm1d(self.bottleneck_size)
        self.adain2 = AdaptiveInstanceNorm1d(self.bottleneck_size // 2)
        self.adain3 = AdaptiveInstanceNorm1d(self.bottleneck_size // 4)

        self.bn1 = torch.nn.BatchNorm1d(
            self.bottleneck_size
        )  # default with Learnable Parameters
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)


    def forward(self, x):
        x = F.relu(self.bn1(self.adain1(self.conv1(x))))
        x = F.relu(self.bn2(self.adain2(self.conv2(x))))
        x = F.relu(self.bn3(self.adain3(self.conv3(x))))
        x = self.th(self.conv4(x))
        return x
    
def get_num_adain_mean_or_std(model):
    """
    input:
    - model: nn.module
    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += m.num_features
    return num_adain_params

def assign_adain_means(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module
    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            m.bias = mean.contiguous().view(-1)
            if adain_params.size(1) > m.num_features:
                adain_params = adain_params[:, m.num_features :]

def assign_adain_deviations(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module
    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            std = adain_params[:, : m.num_features]
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > m.num_features:
                adain_params = adain_params[:, m.num_features :]
                
class PointMorphing(nn.Module):
    def __init__(self, in_channel, step, hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step
        # to do
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.dec = GridDecoder(2, hidden_dim)

        # MLP to generate AdaIN parameters
        self.mlp_global = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, get_num_adain_mean_or_std(self.dec)),
        )
        self.mlp_local = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, get_num_adain_mean_or_std(self.dec)),
        )

    def forward(self, x, q):
        
        num_sample = self.step * self.step
        bs = x.size(0) 
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device) # b 2 n

        adain_deviation = self.mlp_global(x)
        adain_mean = self.mlp_local(q)
        assign_adain_deviations(adain_deviation, self.dec)
        assign_adain_means(adain_mean, self.dec)
        
        fd = self.dec(seed)
        return fd
