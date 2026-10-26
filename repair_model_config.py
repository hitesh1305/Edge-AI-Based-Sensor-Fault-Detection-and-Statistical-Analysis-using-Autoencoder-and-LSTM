
import json
import zipfile
from pathlib import Path

import tensorflow as tf

source_dir = Path("output/models")
repair_dir = source_dir / "compat_test"
repair_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("autoencoder/autoencoder.keras", "autoencoder_compat.keras"),
    ("lstm_predictor/lstm_rul_predictor.keras", "lstm_compat.keras"),
]

def clean_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        obj.pop("input_axes", None)
        obj.pop("output_axes", None)

        for value in obj.values():
            clean_config(value)

    elif isinstance(obj, list):
        for value in obj:
            clean_config(value)

for relative_path, output_name in models:
    source = source_dir / relative_path
    destination = repair_dir / output_name

    print(f"\nPreparing copy: {source}")

    with zipfile.ZipFile(source, "r") as zin:
        with zipfile.ZipFile(destination, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                if item.filename == "config.json":
                    config = json.loads(data)
                    clean_config(config)
                    data = json.dumps(config).encode("utf-8")

                zout.writestr(item, data)

    print("Testing:", destination)

    try:
        model = tf.keras.models.load_model(
            destination,
            compile=False,
            safe_mode=True,
        )
        print("SUCCESS")
        print("Input:", model.input_shape)
        print("Output:", model.output_shape)

    except Exception as exc:
        print("FAILED:", type(exc).__name__, str(exc)[:2500])
