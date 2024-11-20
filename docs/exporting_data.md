# Exporting Data
## What is Data Exporting?

An exporter enables individual samples from an Armory evaluation to be saved
for inspection, visualization, processing, etc.

Task-specific exporters typically include all metadata for the sample
(e.g. ground truth targets, model predictions) and a task-specific representation
of the input data. For example, exporters for computer vision tasks will export
the input data as images.

Exporters use a criteria function to determine which batches or samples are exported.
The `armory.export.criteria` module provides various criteria implementations.

### How to Export Data

Here is an image classification example where samples and targets will be exported every 5 batches.

```python
import armory.evaluation
import armory.export.criteria

evaluation = armory.evaluation.Evaluation(
    name=f"food101-classification",
    description=f"Image classification of food-101",
    author="TwoSix",
)

export_every_n_batches=5
every_n = armory.export.criteria.every_n_batches(export_every_n_batches)
exporter = armory.export.image_classification.ImageClassificationExporter(criterion=every_n)

evaluation.use_exporters(exporter)
```

### Export Criteria

Armory-Library allows users to define criteria providing fine-grained control over the selection
of batches and samples within a batch for export.

* Batch selection using `every_n_batches` and `first_n_batches`;
* Sample selection using `every_n_samples`, `first_n_samples` and `samples`;
* Batch subselection using `every_n_samples_of_batch` and `first_n_samples_of_batch`;
* Sample selection over metric values using `when_metric_eq|lt|gt`, `when_metric_isclose` and `when_metric_in`; and
* Boolean operators over criteria using `all_satisfied`, `any_satisfied` and `not_satisfied`.

### Explainable AI

Armory-Library provides several exporters for Explainable AI that produce
[saliency maps](https://en.wikipedia.org/wiki/Saliency_map) for image classifiers
and detectors. A saliency map is a visual representation that highlights
the most important regions in an image for classification or object detection. 

#### Captum

Armory-Library uses [Captum](https://captum.ai/) to export feature attributions using
the [Integrated Gradients](https://arxiv.org/abs/1703.01365) algorithm.

```python
import armory.export

saliency_n_steps = 50
saliency_classes = [6, 23]
is_saliency_class = armory.export.criteria.when_metric_in(
    armory.export.criteria.batch_targets(),
    saliency_classes,
)

exporter = armory.export.captum.CaptumImageClassificationExporter(
    model=model,
    criterion=is_saliency_class,
    n_steps=saliency_n_steps,
)
```

#### XAITK

Armory-Library provides an exporter for the [Explainable AI Toolkit (XAITK)](https://xaitk.org/) that generates saliency
maps for image classifiers.

```python
import armory.export

saliency_classes = [6, 23]
is_saliency_class = armory.export.criteria.when_metric_in(
    armory.export.criteria.batch_targets(),
    saliency_classes,
)

armory.export.xaitksaliency.XaitkSaliencyBlackboxImageClassificationExporter(
    name="slidingwindow",
    model=model,
    classes=saliency_classes,
    criterion=is_saliency_class,
)
```

#### D-RISE

Armory-Library supports [D-RISE](https://arxiv.org/abs/2006.03204) saliency maps that generate visual explanations
for the predictions of object detectors.

```python
import armory.export

export_every_n_batches=5
every_n = armory.export.criteria.every_n_batches(export_every_n_batches)

armory.export.drise.DRiseSaliencyObjectDetectionExporter(
    model=model,
    criterion=every_n,
    num_classes=1,
    num_masks=10,
)
```