"""
Example Armory evaluation of DeBERTa on the BoolQ dataset.
"""

from typing import Optional

import datasets
import litgpt
import torchmetrics.classification

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
import armory.logging
import armory.metric
import armory.metrics.compute
import armory.model.llm
import armory.track
import armory.utils


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Evaluate DeBERTa on the BoolQ dataset",
        batch_size=2,
        export_every_n_batches=5,
        num_batches=5,
    )
    return parser.parse_args()


def load_model():
    litgpt_model = armory.track.track_params(litgpt.LLM.load)(
        model="meta-llama/Meta-Llama-3.1-8B"
        # model="microsoft/phi-2"
    )

    armory_model = armory.model.llm.LitGPT(
        name="Llama 3.2 1B",
        model=litgpt_model,
        # static_context="System: You are a helpful AI assistant designed to respond only with the words 'true' or 'false' and no further explanation to the user's statement.\nUser:",
        static_context="You are a helpful AI assistant designed to respond 'true' or 'false' to the user's statement.\nUser:",
    )

    return armory_model


def transform(sample):
    sample["question"] = [q + "?" for q in sample["question"]]
    sample["answer"] = [int(a) for a in sample["answer"]]
    return sample


def load_dataset(batch_size: int, shuffle: bool, seed: Optional[int] = None):
    """Load BoolQ dataset from HuggingFace"""

    hf_dataset = armory.track.track_params(datasets.load_dataset)(
        path="google/boolq", split="validation"
    )
    assert isinstance(hf_dataset, datasets.Dataset)
    hf_dataset.set_transform(transform)

    dataloader = armory.dataset.TextClassificationDataLoader(
        hf_dataset,
        inputs_key="question",
        # context_key="passage",
        targets_key="answer",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    dataset = armory.evaluation.Dataset(
        name="boolq",
        dataloader=dataloader,
    )

    return dataset


def create_metrics():
    """Create evaluation metrics"""
    return {
        "accuracy": armory.metric.PredictionMetric(
            torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        ),
    }


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    profiler = armory.metrics.compute.BasicProfiler()
    evaluation = armory.evaluation.Evaluation(
        name="boolq-deberta",
        description="Question answering on BoolQ with DeBERTa",
        author="TwoSix",
    )

    # Model
    with evaluation.autotrack():
        model = load_model()
    evaluation.use_model(model)

    # Dataset
    with evaluation.autotrack():
        dataset = load_dataset(batch_size, shuffle, seed)
    evaluation.use_dataset(dataset)

    # Metrics/Exporters
    # evaluation.use_metrics(create_metrics())

    # Chains
    with evaluation.add_chain("benign"):
        pass

    engine = armory.engine.EvaluationEngine(
        evaluation,
        profiler=profiler,
        limit_test_batches=num_batches,
    )
    results = engine.run()

    if results:
        for chain_name, chain_results in results.children.items():
            chain_results.metrics.table(title=f"{chain_name} Metrics")


if __name__ == "__main__":
    armory.logging.configure_logging()
    main(**vars(parse_cli_args()))
