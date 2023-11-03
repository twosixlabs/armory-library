# from pprint import pprint

import datasets

# import torch.nn as nn
import evaluate
import numpy as np

# from torchmetrics.classification import Accuracy
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (  # BeitImageProcessor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

# CHECKPOINT = "microsoft/beit-base-patch16-224-pt22k-ft22k"
CHECKPOINT = "google/vit-base-patch16-224-in21k"


# class Finetuned(nn.Module):
#     def __init__(self, base_model):
#         super().__init__()
#         self.base_model = base_model
#         self.linear = nn.Linear(1000, 10)

#     def forward(self, *args, **kwargs):
#         result = self.base_model(*args, **kwargs)
#         result = self.linear(result)
#         return result


if __name__ == "__main__":
    dataset = datasets.load_dataset("honaker/eurosat_dataset", split="train[:2000]")
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.train_test_split(test_size=0.2)

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)

    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    # model = Finetuned(model)

    metric = evaluate.load("accuracy")

    # metric = Accuracy(task="multiclass", num_classes=len(label2id))
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="finetuned_eurosat",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=0.001,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("finetuned_eurosat_final")
