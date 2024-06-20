# Evaluation Results

The results of an evaluation are the result of executing an evaluation engine,
or by reconstruction from the data exported to MLFlow.

```python
from armory.results import EvaluationResults

results = engine.run()  # assuming `engine` has been defined elsewhere
# or
results = EvaluationResults.for_run(some_run_id)
# or
results = EvaluationResults.for_last_run(experiment_name=some_exp_name)
```

## Chain Results

The results object from an engine run represents the results of the parent
evaluation, with each chain of the evaluation being a nested, or child, result
object.

```python
with evaluation.add_chain("benign"):
    ...

with evaluation.add_chain("attack"):
    ...

...
results = engine.run()

benign_results = results.children["benign"]
attack_results = results.children["attack"]
```

## Result Properties

The details, parameters, tags, and metrics for each chain may be retrieved in a
structured, dictionary form or displayed as a `rich` or HTML table.

```python
# Dictionary-like access
benign_results.details["run_name"]
benign_results.params["batch_size"]
benign_results.metrics["accuracy"]

# Using rich to print to the console
benign_results.details.table()
benign_results.params.table()
benign_results.tags.table()
benign_results.metrics.table()
benign_results.system_metrics.table()

# Creating HTML tables (i.e., for a notebook)
benign_results.details.plot()
benign_results.params.plot()
benign_results.tags.plot()
benign_results.metrics.plot()
benign_results.system_metrics.plot()
```

## Artifacts

All artifacts attached to the run are available through the results object. If
using a remote MLFlow server, the files are not downloaded until they are read.
Files may be conveniently read or parsed as JSON or PIL images.

```python
# raw file contents (bytes)
benign_results.artifacts["profiler_results.txt"].data

# as a JSON object
benign_results.artifacts["metrics/confusion_matrix.txt"].json

# as a PIL image
benign_results.artifacts["exports/00000/00/input.png"].image
```

## Batch & Sample Artifacts

Structured access to batches and samples is provided, to avoid the need to
construct paths to artifact files like the above example.

```python
benign_results.batch(0).sample(0)["input.png"]
# is the same as
# benign_results.artifacts["exports/00000/00/input.png"]
```

Additionally, access to a sample's exported `metadata.txt` JSON file is
simplified:

```python
benign_results.batch(0).sample(0).metadata["targets"]
# is the same as
# benign_results.artifacts["exports/00000/00/metadata.txt"].json["targets"]
```

## More Examples

See the following notebooks in the `examples/notebooks` folder for additional
demonstrations of visually plotting the results from an evaluation:
    - `image_classification_food101_saliency_maps.ipynb`
    - `object_detection_license_plates_saliency_maps.ipynb`

::: armory.results