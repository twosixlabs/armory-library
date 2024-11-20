# Experiment Tracking

Armory provides integration with [MLFlow](https://mlflow.org/) to track evaluation
runs and store results of metrics evaluations.

Running the Armory evaluation engine creates an experiment using the
`evaluation.name` if one doesn't already exist. Then a parent run is created to
store any global parameters that aren't chain-specific. Each chain within the
evaluation parent run produces a separate nested run. This nested run will contain all
the chain-specific parameters, metrics, and exports.

The following table summarizes how Armory evaluation components map to records
in MLFlow.

| Armory Component      | MLFlow Record                          |
|-----------------------|----------------------------------------|
| Evaluation            | Experiment                             |
| Evaluation engine run | Parent run                             |
| Evaluation chain run  | Nested run                             |
| Tracked params        | Parent or nested run parameters        |
| Metrics               | Nested run metrics _or_ JSON artifacts |
| Exports               | Nested run artifacts

## Usage

Creation and management of runs in MLFlow is handled automatically by the Armory
`EvaluationEngine`.

### Logging Parameters

To automatically record keyword arguments to any function as parameters,
decorate the function with `armory.track.track_params`.

```python
from armory.track import track_params

@track_params
def load_model(name, batch_size):
    pass

model = load_model(name=..., batch_size=...)
```

To automatically record keyword arguments to a class initializer as parameters,
decorate the class with `armory.track.track_init_params`.

```python
from armory.track import track_init_params

@track_init_params
class TheDataset:
    def __init__(self, batch_size):
        pass

dataset = TheDataset(batch_size=...)
```

For third-party functions or classes that do not have the decorator already
applied, use the `track_call` utility function.

```python
from armory.track import track_call

model = track_call(load_model, name=..., batch_size=...)
dataset = track_call(TheDataset, batch_size=...)
```

`track_call` will invoke the function or class initializer given as the first
positional argument, forward all following arguments to the function or class
and record the keyword arguments as parameters.

Additional parameters may be recorded manually using the
`armory.track.track_param` function before the evaluation is run.

```python
from armory.track import track_param

track_param("batch_size", 16)
```

### Tracking Contexts

By default, tracked parameters are recorded in a global context. When
multiple evaluations are executed in a single process, one should take care
with the parameters being recorded. Additionally, all globally recorded
parameters are only associated with the evaluation run's parent run in MLFlow.

The primary way to automatically address these scoping concerns is to use the
evaluation's `autotrack` and `add_chain` contexts.

During an `add_chain` context, all parameters recorded with `track_call`,
`track_params`, or `track_init_params` are scoped to that chain. As a
convenience, the `track_call` function is available as a method on the context's
chain object.

```python
with evaluation.add_chain(...) as chain:
    chain.use_dataset(
        chain.track_call(TheDataset, batch_size=...)
    )
    chain.use_model(
        chain.track_call(load_model, name=..., batch_size=...)
    )
```

For components that are shared among multiple chains, they should be
instantiated within an `autotrack` context. All parameters recorded with
`track_call`, `track_params`, or `track_init_params` are scoped to instances of
`armory.track.Trackable` created during the `autotrack` context. All Armory
dataset, perturbation, model, metric, and exporter wrappers are `Trackable`
subclasses. When a `Trackable` component is associated with an evaluation chain,
all parameters associated with the `Trackable` are then associated with the
chain. As a convenience, the `track_call` function is provided as the context
object for `autotrack`.

```python
with evaluation.autotrack() as track_call:
    model = track_call(load_model, name=..., batch_size=...)

with evaluation.autotrack() as track_call:
    dataset = track_call(TheDataset, batch_size=...)

# All chains will receive the dataset's tracked parameters
evaluation.use_dataset(dataset)

with evaluation.add_chain(...) as chain:
    # Only this chain will receive the model's tracked parameters
    chain.use_model(model)
```

When a parameter is recorded that has already recorded a value, the newer value
will overwrite the old value. When `track_call`, a function decorated with
`track_params`, or a class decorated with `track_init_params` is invoked, all
old values with the same parameter prefix are removed.

```python
from armory.track import track_call

model = track_call(load_model, name="a", extra=True)

# The parameter `load_model.name` is overwritten, `load_model.extra` is removed
model = track_call(load_model, name="b")
```

Parameters can be manually cleared using the `reset_params` function.

```python
from armory.track import reset_params, track_param

track_param("key", "value")

reset_params()
```

While seldomly needed, the `tracking_context` context manager will create a
scoped session for recording of parameters.

```python
from armory.track import tracking_context, track_param

track_param("global", "value")

with tracking_context():
    # `global` parameter will not be recorded within this context
    track_param("parent", "value")

    with tracking_context(nested=True):
        track_param("child", "value")
        # This context contains both `parent` and `child` params, while the
        # outer context still only has `parent`
```

When the evaluation's `autotrack` and `add_chain` contexts are used properly,
there should be no need to explicitly manage tracking contexts or deal with
parameter overwrites.

### Logging Metrics

`EvaluationEngine.run` will automatically log all results of the evaluation as
metrics in MLFlow.

Additional metrics may be logged manually by resuming the MLFlow session after
the evaluation has been run and calling [`mlflow.log_metric`].

```python
import mlflow

engine = EvaluationEngine(evaluation)
engine.run()
with mlflow.start_run(run_id=engine.chains["..."].run_id):
    mlflow.log_metric("custom_metric", 42)
```

### Logging Artifacts

Artifacts generated by exporters are automatically attached to the appropriate
MLFlow runs.

Additional artifacts may be attached manually by resuming the MLFlow session
after the evaulation has been run and calling [`mlflow.log_artifact`] or
[`mlflow.log_artifacts`].

```python
import mlflow

engine = EvaluationEngine(evaluation)
engine.run()
with mlflow.start_run(run_id=engine.chains["..."].run_id):
    mlflow.log_artifacts("path/to/artifacts")
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

When using a remote MLFlow tracking server, set the `MLFLOW_TRACKING_URI`
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

[MLFlow]: https://mlflow.org/docs/latest/tracking.html
[`mlflow.log_metric`]: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
[`mlflow.log_artifact`]: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact
[`mlflow.log_artifacts`]: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifacts
