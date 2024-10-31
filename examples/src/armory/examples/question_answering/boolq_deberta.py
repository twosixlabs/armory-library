"""
Example Armory evaluation of DeBERTa on the BoolQ dataset.
"""

from typing import Optional

import datasets
import transformers

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
import armory.logging
import armory.metrics.compute
import armory.model.question_answering
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
    """Load DeBERTa model from HuggingFace"""
    transformers.AutoModelForQuestionAnswering
    hf_model = armory.track.track_params(
        transformers.AutoModelForSequenceClassification.from_pretrained
    )(pretrained_model_name_or_path="nfliu/deberta-v3-large_boolq")
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "nfliu/deberta-v3-large_boolq"
    )

    armory_model = armory.model.question_answering.SequenceClassificationTransformer(
        name="deberta",
        model=hf_model,
        tokenizer=hf_tokenizer,
    )

    return armory_model


def load_dataset(batch_size: int, shuffle: bool, seed: Optional[int] = None):
    """Load BoolQ dataset from HuggingFace"""

    hf_dataset = armory.track.track_params(datasets.load_dataset)(
        path="google/boolq", split="validation"
    )
    assert isinstance(hf_dataset, datasets.Dataset)

    dataloader = armory.dataset.TextClassificationDataLoader(
        hf_dataset,
        inputs_key="question",
        context_key="passage",
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


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    # profiler = armory.metrics.compute.BasicProfiler()
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

    from pprint import pprint

    batch = next(iter(dataset.dataloader))
    pprint(batch)
    model.predict(batch)
    pprint(batch)


if __name__ == "__main__":
    armory.logging.configure_logging()
    main(**vars(parse_cli_args()))
