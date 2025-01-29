
import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class ResnetPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        dim (int): input dimensionality (default 3)
    """

    def __init__(self, out_dim, hidden_dim, dim=3, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.use_block2 = kwargs.get('use_block2', False)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim) if self.use_block2 else None
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        if self.use_block2:
            net = self.block_2(net)
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.act(net))

        return c


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

@torch.jit.script
def _lin_cond_fwd(input, weight, rf_weight, bias, rank_weight):
    # input: B x N x F
    # weight: F_out x F_in
    # rf_weight: L x R x F_out * F_in
    # bias: B x F_out
    # rank_weight: L x B x R
    # output: B x N x F_out
    _rf_weight = (rank_weight @ rf_weight).sum(0) # B * f_out, f_in
    weight = weight[None] + _rf_weight.view(rank_weight.shape[1], weight.shape[0], weight.shape[1]) # B, f_out, f_in
    return (weight @ input.permute(0, 2, 1) + bias.view(1, -1, 1)).permute(0, 2, 1) # B, S, F_out

@torch.jit.script
def _lin_cond_fwd_part(input, weight, rf_weight, bias, rank_weight, part_weight, part_weight2rank, ind):
    # input: B x N x F
    # weight: F_out x F_in
    # rf_weight: L x R x F_out * F_in
    # bias: F_out
    # rank_weight: L x B x R
    # part_weight: R x F_out * F_in
    # part_weight2rank: K x R
    # ind: B (mapping from batch ind to K ind)
    # output: B x N x F_out
    _rf_weight = (rank_weight @ rf_weight).sum(0) # B, f_out*f_in
    _part_weight = (part_weight2rank @ part_weight)[ind] # B, f_out*f_in
    _rf_weight = _part_weight + _rf_weight
    weight = weight[None] + _rf_weight.view(rank_weight.shape[1], weight.shape[0], weight.shape[1]) # B, f_out, f_in
    return (weight @ input.permute(0, 2, 1) + bias.view(1, -1, 1)).permute(0, 2, 1) # B, S, F_out

class CondResFieldsLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
            rf_kwargs={},
        ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.rank = rf_kwargs.get('rank', 0)
        if self.rank > 0:
            self.ind_dim = rf_kwargs.get('ind_dim', None)
            self.part_rank = rf_kwargs.get('part_rank', 10)

            self.cond_dim = rf_kwargs.get('cond_dim', 16)
            if isinstance(self.cond_dim, int):
                self.cond_dim = [self.cond_dim]
            
            cond2weight = []
            for i, cond_dim in enumerate(self.cond_dim):
                cond2weight.append(torch.nn.Linear(cond_dim, self.rank))
            self.cond2weight = torch.nn.ModuleList(cond2weight)

            self.register_parameter('rf_weight', torch.nn.Parameter(0.001*torch.randn(len(self.cond_dim), self.rank, out_features*in_features))) # L,rank, f_out*f_in
            if self.ind_dim is not None:
                self.register_parameter('part_weight2rank', torch.nn.Parameter(0.001*torch.randn(self.ind_dim, self.part_rank))) # K,rank
                self.register_parameter('part_weight', torch.nn.Parameter(0.001*torch.randn(self.part_rank, out_features*in_features))) # L,rank, f_out*f_in

            # self.register_parameter(f'rf_weight_{i}', torch.nn.Parameter(0.001*torch.randn(self.rank, out_features*in_features)))
            # self.cond2weight = torch.nn.Linear(self.cond_dim, self.rank)

    def _get_cond_weight(self, cond: list):
        # cond: B x C
        # ind: B
        # cond_weight: L x B x rank
        cond_weight = []
        for i in range(len(self.cond_dim)):
            cond_weight.append(self.cond2weight[i](cond[i]))
        cond_weight = torch.stack(cond_weight)
        return cond_weight

    def forward(self, input, cond=None, ind=None):
        # input: B x N x F
        # cond: B x C
        # output: B x N x F
        if self.rank > 0 and cond is not None:
            if not isinstance(cond, list):
                cond = [cond]
            cond_weight = self._get_cond_weight(cond) # B, C -> L, B, capacity
            if ind is not None:
                return _lin_cond_fwd_part(input, self.weight, self.rf_weight, self.bias, cond_weight, self.part_weight, self.part_weight2rank, ind)
            return _lin_cond_fwd(input, self.weight, self.rf_weight, self.bias, cond_weight)
        else:
            return (self.weight @ input.permute(0, 2, 1) + self.bias.view(1, -1, 1)).permute(0, 2, 1) # B, S, F_out
            # return F.linear(input, self.weight, self.bias, out=None)
            return super().forward(input)
            return _lin_fwd(input, self.weight, self.bias)

@torch.jit.script
def _lin_fwd(input, weight, bias):
    return F.linear(input, weight, bias)

class ImplicitNet(nn.Module):
    # adapted from IGR

    def __init__(
            self,
            d_in,
            d_out,
            dims,
            skip_in=(),
            geometric_init=True,
            radius_init=1,
            beta=100,
            rf_kwargs={},
            multires=0,
            **kwargs
    ):
        super().__init__()
        self.multires = multires
        if self.multires > 0:
            embed_fn, in_features = get_embedder(self.multires, input_dims=3)
            self.embed_fn = lambda x: embed_fn(x, alpha_ratio=1.0)
            d_in = d_in - 3 + in_features

        dims = [d_in] + dims + [d_out]

        self.d_out = d_out
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):
            in_dim, out_dim = dims[layer], dims[layer + 1]
            if layer in skip_in:
                in_dim += d_in

            lin = CondResFieldsLinear(in_dim, out_dim, rf_kwargs=rf_kwargs)

            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()
        # import pdb; pdb.set_trace()
        # print(self)

    def forward(self, input, cond=None, ind=None):
        if self.multires > 0:
            input = torch.cat([self.embed_fn(input[..., :3]), input[..., 3:]], -1)
        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x, cond=cond, ind=ind)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) * math.pi
        else:
            freq_bands = torch.linspace(2.**0.*math.pi, 2.**max_freq*math.pi, self.num_freqs)

        self.num_fns = len(self.kwargs['periodic_fns'])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs, alpha_ratio=0.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        start = 0
        if self.include_input:
            start = 1
        for i in range(self.num_freqs):
            _dec = (1.-math.cos(math.pi*(max(min(alpha_ratio*self.num_freqs-i, 1.), 0.)))) * .5
            output[..., (self.num_fns*i+start)*self.input_dims:(self.num_fns*(i+1)+start)*self.input_dims] *= _dec
        return output


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, alpha_ratio, eo=embedder_obj): return eo.embed(x, alpha_ratio)
    return embed, embedder_obj.out_dim
