# Sample Exporters

An exporter enables individual samples from an Armory evaluation to be saved
for post-analysis, demonstration, etc.

Task-specific exporters typically include all metadata for the sample (e.g.
ground truth targets, model predictions) and a task-specific representation of
the input data. For example, exporters for computer vision tasks will export the
input data as images.

Exporters use a criteria function to determine which batches or samples are
exported. The `armory.export.criteria` module provides various criteria
implementations.

::: armory.export.base

::: armory.export.image_classification

::: armory.export.object_detection

::: armory.export.criteria