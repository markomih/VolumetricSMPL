# VolumetricSMPL

## Description
VolumetricSMPL is an extension of the SMPL body model that incorporates a volumetric (signed distance field, SDF) representation. This enables seamless interaction with 3D geometries, such as scenes, objects, and other humans.

## Installation
Ensure that PyTorch and PyTorch3D are installed with GPU support. Then, install VolumetricSMPL via:
```bash
pip install VolumetricSMPL
```

# Usage
VolumetricSMPL extends the interface of the [SMPL-X package](https://github.com/vchoutas/smplx) by attaching a volumetric representation to the body model. This allows for querying signed distance fields for arbitrary points and accessing collision loss terms.

A more detailed tutorial is available [here](https://github.com/markomih/VolumetricSMPL_applications/), demonstrating how to integrate VolumetricSMPL into applications requiring human-scene, human-object, and human-human interactions.

# Example Usage
```python
import smplx
from VolumetricSMPL import attach_volume

# Create a SMPL body and extend it with volumetric functionalities (supports SMPL, SMPLH, and SMPL-X)
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

# Pretrained Models
Pretrained models are automatically fetched and loaded. They can also be found in the `dev` branch inside the `./models` directory.

# Contact
For questions, please contact Marko Mihajlovic (_markomih@ethz.ch_) or open an issue on GitHub. 
