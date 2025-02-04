import torch
import smplx
from .volumetric_smpl import VolumetricSMPL
from .winding_numbers import winding_numbers

def attach_volume(parametric_body: smplx.SMPL, pretrained=True, device=None):
    if parametric_body.name() == 'MANO':
        cfg = {'rf_kwargs': {'rank': 10}, 'decoder_dims': [64, 64, 64, 64, 64, 64], 'decoder_multires': 2, 'decoder_skip_in': [3]}
    else:
        cfg = {'rf_kwargs': {'rank': 80}, 'decoder_dims': [64, 64, 64, 64, 64, 64], 'decoder_multires': 2, 'decoder_skip_in': [3]}
    volumetric_body = VolumetricSMPL(parametric_body, cfg)
    setattr(parametric_body, 'volume', volumetric_body)
    if pretrained:
        model_type = volumetric_body.model_type
        gender = parametric_body.gender
        if parametric_body.name() == 'MANO':
            side = 'right' if parametric_body.is_rhand else 'left'
            checkpoint = f'https://github.com/markomih/VolumetricSMPL/blob/dev/models/VolumetricSMPL_{model_type}_{gender}_{side}.ckpt?raw=true'
        else:
            checkpoint = f'https://github.com/markomih/VolumetricSMPL/blob/dev/models/VolumetricSMPL_{model_type}_{gender}.ckpt?raw=true'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        volumetric_body.load_state_dict(state_dict['state_dict'])
    if device is not None:
        parametric_body = parametric_body.to(device=device)

    # overwrite smpl functions
    def reset_params(self, **params_dict) -> None:
        with torch.no_grad():
            for param_name, param in self.named_parameters():
                if 'volume' in param_name:  # disable reset of volume parameters
                    continue
                if param_name in params_dict:
                    param[:] = torch.tensor(params_dict[param_name])
                else:
                    param.fill_(0)
    setattr(parametric_body, 'reset_params', lambda **x: reset_params(parametric_body, **x))
    
    return parametric_body

__all__ = [
    attach_volume,
]