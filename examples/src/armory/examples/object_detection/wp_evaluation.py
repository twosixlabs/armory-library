
from armory.model.object_detection.yolov4_object_detector import YoloV4ObjectDetector  # type:ignore
import sys

import wp_yolov4 as wp_model  # type:ignore

import armory.utils  # type:ignore
import armory.evaluation  # type:ignore
import armory.engine  # type:ignore

from pprint import pprint
from pytorch_yolo import darknet2pytorch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/encrypted/jcronay/gard-wp/')

cfg_path = '/encrypted/jcronay/gard-wp/data/wp_models/yolov4-custom_wp.cfg'
weights_path = '/encrypted/jcronay/gard-wp/data/wp_models/yolov4-custom_final_plus.weights'
torch_path = '/encrypted/jcronay/gard-wp/data/wp_models/pth_files/yolov4-custom_final_plus.pth'

batch_size = 2
num_batches = 2

_, model_infer = wp_model.load_model(
    cfg_path, is_weights_file=True, weights_path=weights_path)

armory_model = YoloV4ObjectDetector(
    name="YOLOv4",
    model=model_infer
)

opts_benign_data = {
    "data_dir": "/encrypted/jcronay/gard-wp/data/hallucination_dataset_benign",
    "width": 78,
    "height": 34,
    "batch_size": batch_size
}

opts_adv_data = {
    "data_dir": "/encrypted/jcronay/gard-wp/data/hallucination_dataset_adv",
    "width": 78,
    "height": 34,
    "batch_size": batch_size
}

opts_patch = {
    "x_patch_dim": 36,
    "y_patch_dim": 48
}
benign_dataset = wp_model.load_dataset(opts_benign_data, opts_patch)
adv_dataset = wp_model.load_dataset(opts_adv_data, opts_patch)
# attack = wp_model.create_attack(art_estimator)
metrics = wp_model.create_metrics()
exporters = wp_model.create_exporters(export_every_n_batches=1)

evaluation = armory.evaluation.Evaluation(
    name="object-detection-wp-model",
    description="Object detection of the WP model.",
    author="TwoSix",
    dataset=adv_dataset,
    model=armory_model,
    perturbations={
        "benign": [],
        "attack": []
    },
    metrics=metrics,
    exporters=exporters,
)


engine = armory.engine.EvaluationEngine(
    evaluation,
    limit_test_batches=num_batches,
    devices=1
)

results = engine.run()

pprint(results)
