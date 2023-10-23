# Experiment Tracking

Armory provides integration with [MLFlow] to provide tracking of runs (i.e.,
evaluations) within an experiment.

## Usage

The primary interface is the `track_params` and `track_init_params` function
decorators. The Armory `EvaluationEngine` automatically handles the creating and
closing of the MLFlow run session.

### Logging Parameters

To automatically record keyword arguments to any function as parameters,
decorate the function with `charmory.track.track_params`.

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

To automatically record keyword arguments to a class initializer as parameters,
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

Additional parameters may be recorded manually using the
`charmory.track.track_param` function before the evaluation is run.

```python
from charmory.track import track_param

track_param("batch_size", 16)
engine = EvaluationEngine(task)
engine.run()
```

### Logging Metrics

`EvaluationEngine.run` will automatically log all results of the evaluation as
metrics in MLFlow.

Additional metrics may be logged manually by resuming the MLFlow session after
the evaluation has been run and calling [`mlflow.log_metric`].

```python
import mlflow

engine = EvaluationEngine(task)
engine.run()
with mlflow.start_run(run_id=engine.run_id):
    mlflow.log_metric("custom_metric", 42)
```

### Logging Artifacts

Currently, Armory does not automaically log any artifacts with MLFlow. However,
you may log artifacts manually by resuming the MLFlow session after the
evaulation has been run and calling [`mlflow.log_artifact`] or
[`mlflow.log_artifacts`].

```python
import mlflow

engine = EvaluationEngine(task)
engine.run()
with mlflow.start_run(run_id=engine.run_id):
    mlflow.log_artifacts("path/to/artifacts")
```

### Tracking Contexts

By default, all tracked parameters are recorded in a global context. When
multiple evaluations are executed in a single process, one should take care
with the parameters being recorded.

When a parameter is recorded that had already has a recorded value, the newer
value will overwrite the old value. When the `track_params` or
`track_init_params` decorators are used, all old values with the parameter
prefix are removed.

```python
from charmory.track import track_params

model = track_params(load_model)(name="a", extra=True)

# The parameter `load_model.name` is overwritten, `load_model.extra` is removed
model = track_params(load_model)(name="b")
```

Parameters can be manually cleared using the `reset_params` function.

```python
from charmory.track import reset_params, track_param

track_param("key", "value")

reset_params()
```

Alternatively, the `tracking_context` context manager will create a scoped
session for recording of parameters.

```python
from charmory.track import tracking_context, track_param

track_param("global", "value")

with tracking_context():
    # `global` parameter will not be recorded within this context
    track_param("parent", "value")

    with tracking_context(nested=True):
        track_param("child", "value")
        # This context contains both `parent` and `child` params, while the
        # outer context still only has `parent`
```

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
[`mlflow.log_artifact`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact)
[`mlflow.log_artifacts`]: (https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifacts)