"""convenient pre-fabricated "canned" armory evaluation experiments"""

from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    MetaData,
    Metric,
    Model,
    Scenario,
    SysConfig,
)


def cifar10_baseline() -> Evaluation:
    return Evaluation(
        _metadata=MetaData(
            name="cifar_baseline",
            description="Baseline cifar10 image classification",
            author="msw@example.com",
        ),
        model=Model(
            function="armory.baseline_models.pytorch.cifar:get_art_model",
            model_kwargs={},
            wrapper_kwargs={},
            weights_file=None,
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        scenario=Scenario(
            function="armory.scenarios.image_classification:ImageClassificationTask",
            kwargs={},
        ),
        dataset=Dataset(
            function="armory.data.datasets:cifar10", framework="numpy", batch_size=64
        ),
        attack=Attack(
            function="art.attacks.evasion:ProjectedGradientDescent",
            kwargs={
                "batch_size": 1,
                "eps": 0.031,
                "eps_step": 0.007,
                "max_iter": 20,
                "num_random_init": 1,
                "random_eps": False,
                "targeted": False,
                "verbose": False,
            },
            knowledge="white",
            use_label=True,
            type=None,
        ),
        defense=None,
        metric=Metric(
            profiler_type="basic",
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
        sysconfig=SysConfig(gpus=["all"], use_gpu=True),
    )


# def show_mlflow_experiement(experiment_id):
#     experiment = mlflow.get_experiment(experiment_id)
#     print(f"Experiment: {experiment.name}")
#     print(f"tags: {experiment.tags}")
#     print(f"Experiment ID: {experiment.experiment_id}")
#     print(f"Artifact Location: {experiment.artifact_location}")
#     print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
#     print(f"Creation Time: {experiment.creation_time}")

#     def run(self):
#         """fake an evaluation to demonstrate mlflow tracking."""
#         metadata = self.evaluation._metadata
#         log.info("Starting mlflow run:")
#         show_mlflow_experiement(self.experiment_id)
#         self.active_run = mlflow.start_run(
#             experiment_id=self.experiment_id,
#             description=metadata.description,
#             tags={
#                 "author": self.evaluation._metadata.author,
#             },
#         )

#         # fake variable epsilon and results
#         import random

#         epsilon = random.random()
#         result = {"benign": epsilon, "adversarial": 1 - epsilon}
#         self.evaluation.attack.kwargs["eps"] = epsilon

#         for key, value in self.evaluation.flatten():
#             if key.startswith("_metadata."):
#                 continue
#             mlflow.log_param(key, value)

#         for k, v in result.items():
#             mlflow.log_metric(k, v)

#         mlflow.end_run()
#         return result

# # TODO: Integrate logic into demo script above. -CW
# # metadata = evaluation._metadata
# # mlexp = mlflow.get_experiment_by_name(metadata.name)
# # if mlexp:
# #     self.experiment_id = mlexp.experiment_id
# #     log.info(f"Experiment {metadata.name} already exists {self.experiment_id}")
# # else:
# #     self.experiment_id = mlflow.create_experiment(metadata.name)
# #     log.info(
# #         f"Creating experiment {self.evaluation._metadata.name} as {self.experiment_id}"
# #     )
