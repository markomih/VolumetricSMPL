# VolumetricSMPL: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions

[![PyPI version](https://badge.fury.io/py/VolumetricSMPL.svg)](https://pypi.org/project/VolumetricSMPL/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Paper-ICCV%202025%20Highlight-brightgreen)](https://arxiv.org/abs/2506.23236) [![Video](https://img.shields.io/badge/Video-YouTube-red)](https://youtu.be/XmY_W_F58cA)

<div align="center">
  <img src="https://markomih.github.io/VolumetricSMPL/assets/teaser.jpeg" alt="VolumetricSMPL Teaser" width="600"/>
</div>

## 🌟 TL;DR

**VolumetricSMPL** is a lightweight, plug-and-play extension for SMPL(-X) models that adds volumetric functionality via Signed Distance Fields (SDFs). With minimal integration—just a single line of code—users gain access to fast and differentiable SDF queries, collision detection, and self-intersection resolution.

## ✨ Key Features

- 🔌 **Single-line integration** with existing SMPL models
- ⚡ **Fast and differentiable** SDF queries
- 🛡️ **Built-in collision detection** and self-intersection resolution
- 🔄 **Compatible** with SMPL, SMPLH, and SMPL-X
- 🎯 **Efficient interaction modeling** for perception and reconstruction tasks

## 📚 Paper & Resources

- **📄 Paper**: [arXiv](https://arxiv.org/abs/2506.23236)
- **🎥 Video**: [YouTube](https://youtu.be/XmY_W_F58cA)
- **🌐 Project Page**: [markomih.github.io/VolumetricSMPL](https://markomih.github.io/VolumetricSMPL)
- **📦 Applications**: [VolumetricSMPL_applications](https://github.com/markomih/VolumetricSMPL_applications)

## 🚀 Quick Start

### Installation

Ensure that PyTorch and PyTorch3D are installed with GPU support. Then install VolumetricSMPL:

```bash
pip install VolumetricSMPL
```

### Basic Usage

Extend an existing [SMPL-X](https://github.com/vchoutas/smplx) model with volumetric functionalities:

```python
import smplx
from VolumetricSMPL import attach_volume

# Create a SMPL body and extend it with volumetric functionalities
# Supports SMPL, SMPLH, and SMPL-X
model = smplx.create(**smpl_parameters)
attach_volume(model)

# Forward pass
smpl_output = model(**smpl_data)  

# Ensure valid SMPL variables (pose parameters, joints, and vertices)
assert model.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

# Access volumetric functionalities
model.volume.query(scan_point_cloud)                 # Query SDF for given points
model.volume.selfpen_loss(smpl_output)               # Compute self-intersection loss
model.volume.collision_loss(smpl_output, scan_point_cloud)  # Compute collisions with external geometries
```

## 📖 Detailed Usage

VolumetricSMPL extends the interface of the [SMPL-X package](https://github.com/vchoutas/smplx) by attaching a volumetric representation to the body model. This allows for:

- **Querying signed distance fields** for arbitrary points
- **Accessing collision loss terms** for optimization
- **Self-intersection detection** and resolution
- **Efficient interaction modeling** with 3D geometries

For further examples and use cases, check out our [Applications repository](https://github.com/markomih/VolumetricSMPL_applications). 


## 📦 Pretrained Models

Pretrained models are automatically fetched and loaded when you first use VolumetricSMPL. They can also be found in the `dev` branch inside the `./models` directory.

## 🔧 Requirements

- Python 3.7+
- PyTorch 
- PyTorch3D
- SMPL-X

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ICCV25:VolumetricSMPL,
   title={{VolumetricSMPL}: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions},
   author={Mihajlovic, Marko and Zhang, Siwei and Li, Gen and Zhao, Kaifeng and M{\"u}ller, Lea and Tang, Siyu},
   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   year={2025}
}
```

## 👥 Authors

- [Marko Mihajlovic](https://markomih.github.io/) (ETH Zurich)
- [Siwei Zhang](https://sanweiliti.github.io/) (ETH Zurich)
- [Gen Li](https://vlg.inf.ethz.ch/team/Gen-Li.html) (ETH Zurich)
- [Kaifeng Zhao](https://zkf1997.github.io/) (ETH Zurich)
- [Lea Müller](https://muelea.github.io/) (UC Berkeley)
- [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html) (ETH Zurich)

## Contact

For questions, please contact [Marko Mihajlovic](mailto:markomih@ethz.ch) or raise an issue on [GitHub](https://github.com/markomih/VolumetricSMPL).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
