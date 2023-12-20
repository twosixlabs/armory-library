# torchvision sample evaluations

This curated dataset exists to allow armory to evaluate multiple, different models
for comparative analysis.

The dataset is a subset of the [ImageNet](http://www.image-net.org/) dataset,
as retrieved from HuggingFace's [datasets](https://huggingface.co/datasets) library
as parquet files and adapted into th vision.datasets.VisionDataset protocol.

The driver program imgntst.py allows a named model from the torchvision image
classification model library (https://pytorch.org/vision/stable/models.html#classification)
to be evaluated against the dataset.

The names of just about every file here could be better.
