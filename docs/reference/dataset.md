# Dataset Ingestion and Adaptation

As Armory requires input data to be highly-structured, several utilities are
provided to ensure proper ingestion of arbitrary datasets.

All datasets used with Armory must comply with a map-like interface that returns
samples as a dictionary. The following classes are provided to assist with
adaptation of external datasets:

- [`ArmoryDataset`](#armory.dataset.ArmoryDataset) supports use of an adapter
  function to modify samples from an underlying dataset
- [`TupleDataset`](#armory.dataset.TupleDataset) supports conversion of
  underlying datasets whose samples are tuples (e.g., those from torchvision)
  into dictionaries

Armory requires that the datasets be used with task-specific data loaders that
will produce highly-structured, self-describing task-specific data batches for
evaluation. The following classes are provided:

- [`ImageClassificationDataLoader`](#armory.dataset.ImageClassificationDataLoader)
- [`ObjectDetectionDataLoader`](#armory.dataset.ObjectDetectionDataLoader)

::: armory.dataset
