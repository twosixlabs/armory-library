"""
Time-consuming imports
"""

from armory.logs import log

log.info(
    "Importing and configuring torch, tensorflow, and art, if available. "
    "This may take some time."
)

# Handle PyTorch / TensorFlow interplay

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass

# From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("Setting tf.config.experimental.set_memory_growth to True on all GPUs")
except RuntimeError:
    log.exception("Import armory before initializing GPU tensors")
    raise
except ImportError:
    pass

# Handle ART configuration by setting the art data
# path if art can be imported in the current environment
try:
    from art import config
    config.set_data_path(os.path.join(HostPaths().saved_model_dir, "art"))
except ImportError:
    pass
