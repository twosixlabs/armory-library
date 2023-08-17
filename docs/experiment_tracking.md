# Experiment Tracking

Armory provides integration with [MLFlow] to provide tracking of runs (i.e.,
evaluations) within an experiment.

Tracking with MLFlow is optional, and must be explicitly enabled.

## Usage

### Enabling Tracking

To enable logging with MLFlow, you must execute the evaluation within a
`charmory.track.track_evaluation` context. All logging of parameters and metrics
must occur while the context is active.

```python
from charmory.track import track_evaluation

with track_evaluation(name="...", description="..."):
    # Load datasets, models, attacks, defenses, etc.
    engine = Engine(evaluation)
    engine.run()
```

### Logging Metrics

When tracking is active, `Engine.run` will automatically log all results of the
evaluation as metrics in MLFlow.

Additional metrics may be logged manually using the [`mlflow.log_metric`]
function.

### Logging Parameters

To automatically log keyword arguments to any function as parameters, decorate
the function with `charmory.track.track_params`.

```python
from charmory.track import track_params

@track_params
def load_model(name, batch_size):
    pass

model = load_model(name=..., batch_size=...)
```

Or for a third-party function that cannot have the decorator already applied,
wrap the function with `track_params` before calling it.

```python
model = track_params(load_model)(name=..., batch_size=...)
```

To automatically log keyword arguments to a class initializer as parameters,
decorate the class with `charmory.track.track_init_params`.

```python
from charmory.track import track_init_params

@track_init_params
class TheDataset:
    def __init__(self, batch_size):
        pass

dataset = TheDataset(batch_size=...)
```

Or for a third-party class that cannot have the decorator already applied,
wrap the class with `track_init_params` before creating an instance of it.

```python
dataset = track_init_params(TheDataset)(batch_size=...)
```

Additional parameters may be logged manually using the [`mlflow.log_param`]
function.

## Logging Artifacts

Currently, Armory does not automaically log any artifacts with MLFlow. However,
you may log artifacts manually using the [`mlflow.log_artifact`] or
[`mlflow.log_artifacts`] functions after the evaluation has been run.

## Tracking Server

### Local

By default, all evaluation tracking will be stored in a local database under
`~/.armory/mlruns`. To launch a local version of the MLFlow server configured to
use this local database, a convenience entrypoint is provided as part of Armory.

```sh
armory-mlflow-server
```

And you can view it at `http://localhost:5000` in your browser.

### Remote

If using a remote MLFlow tracking server, set the `MLFLOW_TRACKING_URI`
environment variable to the tracking server's URI.

```sh
export MLFLOW_TRACKING_URI=https://<mlflow_tracking_uri/
python run_my_evaluation.py
```

If the remote tracking server has authentication enabled, you must also set the
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` environment variables.

```sh
export MLFLOW_TRACKING_URI=https://<mlflow_tracking_uri/
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
python run_my_evaluation.py
```

You may also store your credentials
[in a file](https://mlflow.org/docs/latest/auth/index.html#using-credentials-file).

[MLFlow]: (https://mlflow.org/docs/latest/tracking.html)
[`mlflow.log_metric`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric)
[`mlflow.log_param`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param)
[`mlflow.log_artifact`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact)
[`mlflow.log_artifacts`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifacts)