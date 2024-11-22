"""
Example Armory evaluation of LitGPT Llama 3.
"""

from typing import Optional

import datasets
import litgpt

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

# import torchmetrics.classification


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Evaluate LitGPT Llama on a jailbreak prompt dataset",
        batch_size=2,
        export_every_n_batches=5,
        num_batches=5,
    )
    return parser.parse_args()


def load_model():
    """Load LitGPT Llama model"""
    litgpt_model = armory.track.track_params(litgpt.LLM.load)(
        model="meta-llama/Llama-3.2-1B"
    )

    armory_model = armory.model.llm.LitGPT(
        name="Llama 3.2 1B",
        model=litgpt_model,
    )

    return armory_model


def load_dataset(batch_size: int, shuffle: bool, seed: Optional[int] = None):
    """Load jailbreak dataset from HuggingFace"""

    hf_dataset = armory.track.track_params(datasets.load_dataset)(
        path="TrustAIRLab/in-the-wild-jailbreak-prompts",
        name="jailbreak_2023_05_07",
        split="train",
    )
    assert isinstance(hf_dataset, datasets.Dataset)

    dataloader = armory.dataset.TextPromptDataLoader(
        hf_dataset,
        inputs_key="prompt",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    dataset = armory.evaluation.Dataset(
        name="jailbreak",
        dataloader=dataloader,
    )

    return dataset


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    profiler = armory.metrics.compute.BasicProfiler()
    evaluation = armory.evaluation.Evaluation(
        name="jailbreak-litgpt-llama",
        description="Jailbreaking LitGPT Llama",
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
