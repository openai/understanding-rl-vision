import os
import urllib.request
from setuptools import setup, find_packages

REMOTE_JS_PATH = (
    "https://openaipublic.blob.core.windows.net/rl-clarity/attribution/js/interface.js"
)


def download_js():
    dir_ = os.path.dirname(os.path.realpath(__file__))
    js_dir_path = os.path.join(dir_, "understanding_rl_vision/rl_clarity/js")
    js_path = os.path.join(js_dir_path, "interface.js")
    if not os.path.isfile(js_path):
        if not os.path.exists(js_dir_path):
            os.mkdir(js_dir_path)
        try:
            urllib.request.urlretrieve(REMOTE_JS_PATH, js_path)
        except:
            if os.path.exists(js_path):
                os.remove(js_path)


setup(
    name="understanding-rl-vision",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "mpi4py",
        "baselines",
        "lucid @ git+https://github.com/tensorflow/lucid.git@16a03dee8f99af4cdd89d6b7c1cc913817174c83",
    ],
    extras_require={"envs": ["coinrun", "procgen", "atari-py"]},
)

download_js()
