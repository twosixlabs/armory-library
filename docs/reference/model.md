# Model Ingestion and Adaptation
Models help adapt the input and output data of the evaluation to a user's needs. The pre- and post-adapters can be specified on the base ArmoryModel. The JATIC Image Classiifcation model class has a pre-defined output adapter specific to that task which includes logits, probabilities, and scores. The JATIC Object Detection model also has a pre-defined output adapter which includes boxes, logits, probabilities, and scores. The YOLOS Transformer has pre-defined input and output adapters for HuggingFace transformer YOLOS models. On input, the "labels" property becomes "class_labels". On output, the loss components are returned instead of the predictions since the model is put into training mode during attack generation.

::: charmory.model.base

::: charmory.model.image_classification

::: charmory.model.object_detection
