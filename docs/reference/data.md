# Highly-Structured Data Types

An Armory evaluation pipeline is made up of several stages, with the outputs of
one stage typically being the inputs to the next stage. The format, scaling,
and structure of data used as input or  output to any given stage can vary
according to the exact components used in each stage (e.g., a perturbation may
operate on HWC NumPy image data but the model requires CHW PyTorch tensor image
data). To better support composability of these independent components and avoid
implicit dependencies, Armory requires all data in the evaluation pipeline be
defined using custom self-describing data types.

For example, in computer vision evaluations the image data is defined in an
`Images` type that describes the dimensional format of the data and how the
values are scaled. When a stage component requires image data, it acquires a
copy of the raw image data by providing the desired images data specification
that declares the specific dimension format and value scaling required by that
component. The component does not need to know the format or scaling of the
image data from a prior stage, as the image data is automatically converted to
the required format or scaling.

## Convertible Types

Currently, all self-describing data types support conversion to raw data based
on the following types:

- NumPy arrays
- PyTorch tensors

For example, given a `BoundingBoxes` object instance it is possible to acquire a
version where the boxes, labels, and scores are represented using NumPy arrays
or a version where they are represented using PyTorch tensors.

```python
import numpy as np
import torch
import armory.data

boxes = armory.data.BoundingBoxes(
    boxes=[
        dict(
            boxes=np.ndarray([[1, 2, 3, 4]]),
            labels=np.ndarray([0]),
        ),
    ],
    spec=armory.data.BoundingBoxSpec(
        format=armory.data.BBoxFormat.XYXY,
    ),
)

assert type(boxes.get(armory.data.NumpySpec())[0]["boxes"]) == np.ndarray
assert type(boxes.get(armory.data.TorchSpec())[0]["boxes"]) == torch.Tensor
```

In addition to conversion between raw data types, it is also possible to convert
between formats or scaling of values.

```python
import numpy as np
import armory.data

images = armory.data.Images(
    images=np.random.rand(2, 3, 100, 100),
    spec=armory.data.ImageSpec(
        dim=armory.data.ImageDimensions.CHW,
        scale=armory.data.Scale(
            dtype=armory.data.DataType.FLOAT, max=1.0
        ),
    ),
)

# convert (when necessary) between array dimension formats
assert images.get(
    armory.data.ImageSpec(
        dim=armory.data.ImageDimensions.CHW,
        scale=images.spec.scale,
    )
).shape == (2, 3, 100, 100)
assert images.get(
    armory.data.ImageSpec(
        dim=armory.data.ImageDimensions.HWC,
        scale=images.spec.scale,
    )
).shape == (2, 100, 100, 3)

# convert (when necessary) between value scaling (including normalization)
assert images.get(
    armory.data.ImageSpec(
        scale=armory.data.Scale(dtype=armory.data.DataType.FLOAT, max=1.0),
        dim=images.spec.dim,
    )
).max() == 1.0
assert images.get(
    armory.data.ImageSpec(
        scale=armory.data.Scale(
            dtype=armory.data.DataType.UINT8, 
            max=255
        ),
        dim=images.spec.dim,
    )
).max() == 255

assert images.get(
    armory.data.ImageSpec(
        scale=armory.data.Scale(
            dtype=armory.data.DataType.FLOAT,
            max=1.0,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        dim=images.spec.dim,
    )
).max() == 3.0
```

Currently the following self-describing data types are provided:

- [`Images`](#armory.data.Images)
- [`NDimArray`](#armory.data.NDimArray)
- [`BoundingBoxes`](#armory.data.BoundingBoxes)

## Data Specifications

In order to specify what format, structure, or value constraints a component
requires of the raw data obtained from a self-describing data object, a data
specification can be created and provided to the component.

```python
import armory.data

perturbation_spec = armory.data.NumpyImageSpec(
    dim=armory.data.ImageDimensions.HWC,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0
    ),
)

model_spec = armory.data.TorchImageSpec(
    dim=armory.data.ImageDimensions.CHW,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
)
```

When a component needs to acquire a copy of the raw data, it uses the `get`
function on the self-describing data object and provides the data specification
object that describes the requires of the raw data. If any conversion of the
data is required, it will be performed on the returned copy.

```python
import numpy as np
import torch
import armory.data

images = armory.data.Images(
    images=np.random.rand(2, 3, 100, 100),
    spec=armory.data.ImageSpec(
        dim=armory.data.ImageDimensions.CHW,
        scale=armory.data.Scale(
            dtype=armory.data.DataType.FLOAT, max=1.0
        ),
    ),
)

assert type(images.get(perturbation_spec)) == np.ndarray
assert type(images.get(model_spec)) == torch.Tensor
```

If a component is updating the self-describing data object (e.g. replacing
input with a perturbed version, storing model predictions), it uses the
self describing data object's `set` method and provides a data specification
for the new data.

```python
import numpy as np
import torch

# the perturbation_spec's dimension format is HWC, so the new raw image data
# must be in HWC format
images.set(np.random.rand(2, 100, 100, 3), perturbation_spec)

# the model_spec's dimension format is CHW, so the new raw image data must be in
# CHW format
images.set(np.random.rand(2, 3, 100, 100), model_spec)
```

::: armory.data
