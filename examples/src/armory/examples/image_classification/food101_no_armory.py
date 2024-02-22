"""
Example evaluation of food-101 image classification against projected
gradient descent (PGD) adversarial perturbation
"""

import functools
from pprint import pprint

import art.attacks.evasion
import art.defences.preprocessor
import art.estimators.classification.hugging_face
import datasets
import mlflow
import torch
import torch.nn
import torch.utils.data.dataloader
import torchmetrics
import torchmetrics.classification
import torchvision.transforms
import transformers

mlflow.set_experiment("food101-classification")

with mlflow.start_run(log_system_metrics=True):
    # Model

    hf_model = transformers.AutoModelForImageClassification.from_pretrained(
        pretrained_model_name_or_path="nateraw/food"
    )

    art_classifier = (
        art.estimators.classification.hugging_face.HuggingFaceClassifierPyTorch(
            hf_model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(hf_model.parameters(), lr=0.003),
            input_shape=(3, 224, 224),
            channels_first=True,
            nb_classes=101,
            clip_values=(-1.0, 1.0),
        )
    )

    # Dataset

    def transform(processor, sample):
        """Use the HF image processor and convert from BW To RGB"""
        sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
            "pixel_values"
        ]
        return sample

    hf_dataset = datasets.load_dataset("food101", split="validation")
    assert isinstance(hf_dataset, datasets.Dataset)

    labels = hf_dataset.features["label"].names

    hf_processor = transformers.AutoImageProcessor.from_pretrained("nateraw/food")
    hf_dataset.set_transform(functools.partial(transform, hf_processor))

    dataloader = torch.utils.data.dataloader.DataLoader(
        hf_dataset,
        batch_size=16,
    )

    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    normalize = torchvision.transforms.Normalize(MEAN, STD)

    def unnormalize(data):
        images = data.copy()
        for image in images:
            for c, m, s in zip(image, MEAN, STD):
                c *= s
                c += m
        return images

    # Attack

    pgd = art.attacks.evasion.ProjectedGradientDescent(
        art_classifier,
        batch_size=1,
        eps=0.003,
        eps_step=0.0007,
        max_iter=20,
        num_random_init=1,
        random_eps=False,
        targeted=False,
        verbose=False,
    )

    mlflow.log_params(
        {
            "ProjectedGradientDescent.batch_size": 1,
            "ProjectedGradientDescent.eps": 0.003,
            "ProjectedGradientDescent.eps_step": 0.0007,
            "ProjectedGradientDescent.max_iter": 20,
            "ProjectedGradientDescent.num_random_init": 1,
            "ProjectedGradientDescent.random_eps": False,
            "ProjectedGradientDescent.targeted": False,
        }
    )

    # Defense

    jpeg_compression = art.defences.preprocessor.JpegCompression(
        clip_values=(0.0, 1.0),
        quality=50,
        channels_first=True,
    )

    mlflow.log_params(
        {
            "Jpegcompression.quality": 50,
        }
    )

    # Metrics

    class PerturbationNormMetric(torchmetrics.Metric):
        def __init__(self):
            super().__init__()
            self.distance: torch.Tensor
            self.total: torch.Tensor
            self.add_state(
                "distance",
                default=torch.tensor(0, dtype=torch.float32),
                dist_reduce_fx="sum",
            )
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, natural, perturbed):
            self.distance += torch.norm((natural - perturbed).flatten(), p=torch.inf)
            self.total += torch.tensor(1)

        def compute(self):
            return self.distance.float() / self.total

    benign_accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=101
    )

    attacked_accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=101
    )
    attacked_linf_norm = PerturbationNormMetric()

    defended_accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=101
    )
    defended_linf_norm = PerturbationNormMetric()

    # Evaluation

    num_batches = 0
    for batch in dataloader:
        label = batch["label"]
        image = batch["image"].cpu().numpy()

        # benign
        benign_preds = art_classifier.predict(image)
        benign_accuracy.update(
            torch.as_tensor(benign_preds),
            label,
        )

        # attacked
        attacked_image = pgd.generate(
            x=image,
            y=label.cpu().numpy(),
        )
        attacked_preds = art_classifier.predict(attacked_image)
        attacked_accuracy.update(
            torch.as_tensor(attacked_preds),
            label,
        )
        attacked_linf_norm.update(
            torch.as_tensor(image), torch.as_tensor(attacked_image)
        )

        # defended
        # have to de-normalize the image for JPEG compression, then re-normalize it
        # before invoking the model
        attacked_image_unnormed = unnormalize(attacked_image)
        defended_image_unnormed, _ = jpeg_compression(attacked_image_unnormed)
        defended_image = (
            normalize(torch.as_tensor(defended_image_unnormed)).cpu().numpy()
        )
        defended_preds = art_classifier.predict(defended_image)
        defended_accuracy.update(
            torch.as_tensor(defended_preds),
            label,
        )
        defended_linf_norm.update(
            torch.as_tensor(image), torch.as_tensor(defended_image)
        )

        if (num_batches + 1) % 5 == 0:
            for idx, img in enumerate(unnormalize(image)):
                mlflow.log_image(
                    img.transpose(1, 2, 0),
                    f"batch_{num_batches}_ex_{idx}_benign.png",
                )
                meta = {
                    "targets": label[idx].item(),
                    "predictions": [p.item() for p in benign_preds[idx]],
                }
                mlflow.log_dict(meta, f"batch_{num_batches}_ex_{idx}_benign.txt")

            for idx, img in enumerate(unnormalize(attacked_image)):
                mlflow.log_image(
                    img.transpose(1, 2, 0),
                    f"batch_{num_batches}_ex_{idx}_attacked.png",
                )
                meta = {
                    "targets": label[idx].item(),
                    "predictions": [p.item() for p in attacked_preds[idx]],
                }
                mlflow.log_dict(meta, f"batch_{num_batches}_ex_{idx}_attacked.txt")

            for idx, img in enumerate(unnormalize(defended_image)):
                mlflow.log_image(
                    img.transpose(1, 2, 0),
                    f"batch_{num_batches}_ex_{idx}_defended.png",
                )
                meta = {
                    "targets": label[idx].item(),
                    "predictions": [p.item() for p in defended_preds[idx]],
                }
                mlflow.log_dict(meta, f"batch_{num_batches}_ex_{idx}_defended.txt")

        num_batches += 1
        if num_batches >= 5:
            break

    mlflow.log_metrics(
        {
            "benign/accuracy": benign_accuracy.compute().item(),
            "attacked/accuracy": attacked_accuracy.compute().item(),
            "attacked/linf_norm": attacked_linf_norm.compute().item(),
            "defended/accuracy": defended_accuracy.compute().item(),
            "defended/linf_norm": defended_linf_norm.compute().item(),
        }
    )


pprint(
    {
        "metrics": {
            "benign/accuracy": benign_accuracy.compute(),
            "attacked/accuracy": attacked_accuracy.compute(),
            "attacked/linf_norm": attacked_linf_norm.compute(),
            "defended/accuracy": defended_accuracy.compute(),
            "defended/linf_norm": defended_linf_norm.compute(),
        }
    }
)
