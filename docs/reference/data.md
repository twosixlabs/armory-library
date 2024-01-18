# Dataset Ingestion and Adaptation
A dataset is a collection of images, i.e. the sample set, in a sequence-like structure. This class is used in the initial uploading and manipulation of a dataset and supports converting 2 different types of data into a structure compatible with armory-library:

- Tuples datasets which can be turned into a map
- PyTorch datasets needing to be turned into numpy arrays

::: armory.data
