# Evaluation Tasks
A task should match what the model is intended to do. Right now, the options are image classification and object detection, but more could be supported in the future.

The class holds details of the entire task to include the evaluation configuration, whether to skip the benign and/or attack datasets, an optional adapter to be applied to the inference data prior to exporting to MLflow, and a frequency at which batches will be exported to MLflow, if at all.

The Image Classification task can additionally be created with the total number of classes the model is capable of predicting, and the L-norm order for the perturbation distance metrics.

The Object Detection task can additionally be created with an option to track Mean Average Precision (MAP) metrics per class, with a minimum prediction score for a detection bounding box to be drawn on the exported sample, a maximum intersection-over-union value for non-maximum suppression filtering of detection bounding boxes, and a minimum prediction score, with all predictions lower than this being ignored.

::: armory.tasks.base.BaseEvaluationTask

::: armory.tasks.image_classification.ImageClassificationTask

::: armory.tasks.object_detection.ObjectDetectionTask
