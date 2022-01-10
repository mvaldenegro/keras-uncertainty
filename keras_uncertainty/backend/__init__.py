import sys
import os
import json

CONFIG_FILE = "config.json"
DEFAULT_BACKEND = "tfkeras"
DEFAULT_CONFIG = {
    "backend": DEFAULT_BACKEND
}

ku_base_dir = os.path.expanduser("~")
ku_dir = os.path.join(ku_base_dir, '.keras_unc')

config_path = os.path.expanduser(os.path.join(ku_dir, CONFIG_FILE))
backend = DEFAULT_BACKEND

if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
    except ValueError:
        config = {}

    backend = config.get("backend", DEFAULT_BACKEND)

if not os.path.exists(ku_dir):
    try:
        os.makedirs(ku_dir)
    except IOError:
        pass

if not os.path.exists(config_path):
    try:
        with open(config_path, 'w') as f:
            f.write(json.dumps(DEFAULT_CONFIG, indent=4))
    except IOError:
        pass

if backend == "keras":
    sys.stderr.write("Keras Uncertainty will use standalone Keras backend")

    from .keras_backend import layers, losses, metrics, models, activations, initializers, utils, regularizers
    from .keras_backend import *

if backend == "tfkeras":
    print("Keras Uncertainty will use tensorflow.keras backend")

    from .tfkeras_backend import layers, losses, metrics, models, activations, initializers, utils, regularizers
    from .tfkeras_backend import *

#For now implement einsum for tensorflow only
from tensorflow import einsum

#Expose local copy of conv_utils
from . import conv_utils