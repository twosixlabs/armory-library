# Armory Engines
An engine is the core of the Armory library. There are two different types of engines which the user should choose from based on their overall objective:
- The Evaluation Engine performs model robustness evaluations as pertains to adversarial attacks. Can optionally be recorded in MLflow.
- The Adversarial Dataset Engine creates the adversarial dataset by applying an attack to each sample in the original dataset, outputting the results into a directory. Additional modifications to the samples may also be preformed.

::: armory.engine.EvaluationEngine

::: armory.engine.AdversarialDatasetEngine
