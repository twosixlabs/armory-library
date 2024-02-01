import evaluate
import numpy as np
from torchvision.transforms import (  # RandomResizedCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from armory.evaluation import SysConfig
import armory.examples.image_classification.icon_classifier.icon645 as icon645

checkpoint = "google/vit-base-patch16-224-in21k"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
train_transforms = Compose([Resize(size), ToTensor(), normalize])
eval_transforms = Compose([Resize(size), ToTensor(), normalize])

accuracy = evaluate.load("accuracy")


def train_transform(examples):
    examples["pixel_values"] = [
        train_transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples


def eval_transform(examples):
    examples["image"] = [
        eval_transforms(img.convert("RGB")) for img in examples["image"]
    ]
    return examples


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def output_path(sysconfig: SysConfig):
    return sysconfig.armory_home / "models" / "vit-finetuned-icon645-nocrop-out"


def weights_path(sysconfig: SysConfig):
    return sysconfig.armory_home / "models" / "vit-finetuned-icon645-nocrop"


def finetune(sysconfig: SysConfig = SysConfig()):
    ds = icon645.load_dataset()
    ds = ds.with_transform(train_transform)

    # Use subset of whole dataset
    train_ds = ds["train"].select(range(8000))
    eval_ds = ds["validation"].select(range(2000))

    data_collator = DefaultDataCollator()

    labels = ds["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for idx, label in enumerate(labels):
        label2id[label] = str(idx)
        id2label[str(idx)] = label

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path(sysconfig)),
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=18,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(str(weights_path(sysconfig)))


def load_model(sysconfig: SysConfig = SysConfig()):
    return AutoModelForImageClassification.from_pretrained(str(weights_path(sysconfig)))


if __name__ == "__main__":
    finetune()
