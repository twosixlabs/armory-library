# Evaluation Configuration

An [evaluation](#armory.evaluation.Evaluation) completely defines how Armory
will evaluate a model. An evaluation is comprised of chains. Each chain is
independent, and identifies the dataset input, the perturbations that will be
applied to the dataset samples, the model, what metrics will be calculated on
the model outputs, and any exports of samples or predictions.

Chains are defined on an evaluation object using the `add_chain` context.

```python
from armory.evaluation import Evaluation

evaluation = Evaluation(name="...", description="...", author="...")

with evaluation.add_chain("benign") as chain:
    chain.use_dataset(...)
    chain.use_model(...)
    chain.use_metrics(...)
    chain.use_exporters(...)

with evaluation.add_chain("attack") as chain:
    chain.use_dataset(...)
    chain.use_perturbations(...)  # this chain has input perturbations
    chain.use_model(...)
    chain.use_metrics(...)
    chain.use_exporters(...)
```

When a component is shared between chains, it can be declared a default for the
evaluation. If a chain does not specify or override the component inside
`add_chain`, the default component will be applied to the chain.

```python
from armory.evaluation import Evaluation

evaluation = Evaluation(name="...", description="...", author="...")

# Common components
evaluation.use_dataset(...)
evaluation.use_metrics(...)
evaluation.use_exporters(...)
evaluation.use_model(...)

with evaluation.add_chain("benign") as chain:
    pass

with evaluation.add_chain("attack") as chain:
    chain.use_perturbations(...)
```

See the [tracking documentation](../experiment_tracking.md) for additional
information about how to automatically track evaluation parameters when defining
evaluation chains.

::: armory.evaluation
