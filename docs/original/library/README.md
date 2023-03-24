# Armory as a Library

## strategy for decomposition

1. Define the Experiment class which 1:1 maps from config.yaml
2. Pare down the function of Scenario as in #Scenario below
3. Flesh out the Experiment block class (e.g. Attack) with code pulled out of
    Scenario

## Experiment

Experiment = json.load(config.json)

This operation allows us to keep the concept of a config file, but becomes an
argument to Loader/Engine/etc.

    class Experiment:
        load_from_file(path)
        __init__(**kwargs)

See [experiment.py](docs/experiment.py) for a sample implementation of the
various blocks of class Experiment.

## Scenario

Should be the bit of current armory.Scenario which is not related to loading
models, attacks, datasets, etc. This means that we need to method-by-method
figure out what is actually the Scenario and move everything else out of that
God object.
