# Sample Exporters

An exporter enables individual samples from an Armory evaluation to be saved
for post-analysis, demonstration, etc. An export typically includes all metadata
for the sample (e.g. ground truth targets, model predictions) and a
task-specific representation of the input data. For example, exporters for
computer vision tasks will export the input data as images.

::: armory.export.base

::: armory.export.image_classification

::: armory.export.object_detection