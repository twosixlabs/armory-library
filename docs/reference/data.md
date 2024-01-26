# Highly-Structured Data Types

An Armory evaluation pipeline is made up of several stages, with the outputs of
one stage typically being the inputs to the next stage. The format, scaling,
and structure of data used as input or  output to any given stage can vary
according to the exact components used in each stage (e.g., a perturbation may
operate on HWC NumPy image data but the model requires CHW PyTorch tensor image
data). To better support composability of these independent components and avoid
implicit dependencies, Armory requires all data in the evaluation pipeline be
defined using custom highly-structured data types.

For example, in computer vision evaluations the image data is defined in an
`Images` type that describes the dimensional format of the data and how the
values are scaled. When a stage component requires image data, it acquires a
low-level representation of the image data using an accessor that has been
configured with the specific dimension format and value scaling required by that
component. The component does not need to know the format or scaling of the
image data from a prior stage, as the image data is automatically converted to
the required format or scaling by using the accessor.

## Convertible Types

Currently, all highly-structured data types support conversion to one of the
following low-level representational types:

- NumPy arrays
- PyTorch tensors

For example, given a highly-structured `BoundingBoxes` object instance it is
possible to acquire a low-level version where the boxes, labels, and scores are
represented using NumPy arrays or a version where they are represented using
PyTorch tensors.

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
    format=armory.data.BBoxFormat.XYXY,
)

assert type(boxes.to_numpy_boxes()[0]["boxes"]) == np.ndarray
assert type(boxes.to_torch_boxes()[0]["boxes"]) == torch.Tensor
```

In addition to conversion to a low-level representational data type, it is also
possible to convert between formats or scaling of values.

```python
import numpy as np
import armory.data

images = armory.data.Images(
    images=np.random.rand(2, 3, 100, 100),
    dim=armory.data.ImageDimensions.CHW,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0
    ),
)

# convert (when necessary) between array dimension formats
assert images.to_numpy_images(dim=armory.data.ImageDimensions.CHW).shape == (2, 3, 100, 100)
assert images.to_numpy_images(dim=armory.data.ImageDimensions.HWC).shape == (2, 100, 100, 3)

# convert (when necessary) between value scaling (including normalization)
assert images.to_numpy_images(scale=armory.data.Scale(
    dtype=armory.data.DataType.FLOAT, max=1.0
)).max() == 1.0
assert images.to_numpy_images(scale=armory.data.Scale(
    dtype=armory.data.DataType.UINT8, max=255
)).max() == 255
assert images.to_numpy_images(scale=armory.data.Scale(
    dtype=armory.data.DataType.FLOAT, max=1.0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)).max() == 3.0
```

Currently the following highly-structured data types are provided:

- [`Images`](#armory.data.Images)
- [`NDimArray`](#armory.data.NDimArray)
- [`BoundingBoxes`](#armory.data.BoundingBoxes)

## Accessors

In order to specify how a component must acquire a low-level representation from
a highly-structured data object (i.e. whether to invoke a `to_numpy` or a
`to_torch` function, and what conversion arguments are necessary), a
type-specific `Accessor` instance can be created and provided to the component.
The accessor functions similarly to `functools.partial` from the Python standard
library.

```python
import armory.data

perturbation_accessor = armory.data.Images.as_numpy(
    dim=armory.data.ImageDimensions.HWC,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0
    ),
)

model_accessor = armory.data.Images.as_torch(
    dim=armory.data.ImageDimensions.CHW,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
)
```

When a component needs to acquire the low-level representation, it uses the
accessor's `get` function and provides only the highly-structured data object.

```python
import numpy as np
import torch
import armory.data

images = armory.data.Images(
    images=np.random.rand(2, 3, 100, 100),
    dim=armory.data.ImageDimensions.CHW,
    scale=armory.data.Scale(
        dtype=armory.data.DataType.FLOAT, max=1.0
    ),
)

assert type(perturbation_accessor.get(images)) == np.ndarray
assert type(model_accessor.get(images)) == torch.Tensor
```

If a component is updating the highly-structured data object (e.g. replacing
input with a perturbed version, storing model predictions), it uses the
accessor's `set` method.

```python
import numpy as np
import torch

# because the perturbation_accessor was created with a dimension format of HWC,
# it expects the argument to `set` to be in HWC format
perturbation_accessor.set(images, np.random.rand(2, 100, 100, 3))

# because the model_accessor was created with a dimension format of CHW,
# it expects the argument to `set` to be in CHW format
model_accessor.set(images, np.random.rand(2, 3, 100, 100))
```

::: armory.data
