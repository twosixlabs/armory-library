# Utilities
A variety of utilities to aid in evaluation tasks. Includes:

- A custom torchvision transform which converts PIL images to numpy arrays
- A customizable transform which serves as the inverse of torchvision.transforms.Normalize
- The ability to apply a given ART pre- or post-processor defense to a model
- A transform that can be applied to JATIC-wrapped datasets using a preprocessor from a JATIC-wrapped model
- A check for whether a given estimator has any pre- or post-processor defenses applied to it

::: armory.utils