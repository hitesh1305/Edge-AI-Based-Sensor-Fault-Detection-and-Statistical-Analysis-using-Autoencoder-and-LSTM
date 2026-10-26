
import json
import zipfile
import tensorflow as tf
from pathlib import Path
from keras.initializers import GlorotUniform

class CompatibleGlorotUniform(GlorotUniform):
    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        super().__init__(seed=seed, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return cls(**config)

models = [
    "output/models/autoencoder/autoencoder.keras",
    "output/models/lstm_predictor/lstm_rul_predictor.keras",
]

for filename in models:
    print("\nLoading:", filename)
    try:
        model = tf.keras.models.load_model(
            filename,
            compile=False,
            custom_objects={"GlorotUniform": CompatibleGlorotUniform},
            safe_mode=True,
        )
        print("SUCCESS")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:1000])
