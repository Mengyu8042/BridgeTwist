"""MNIST Super Learner external classifier (Curt-Park/handwritten_digit_recognition).

The upstream model definitions use Keras 2.2.x APIs. Inference compatibility
is provided by:
  1. Replacing unused training helper imports with lightweight placeholders.
  2. Patching `keras.applications.mobilenet.DepthwiseConv2D`, which moved.
  3. Patching `keras.optimizers.{SGD,RMSprop,Adam}` to ignore deprecated
     optimizer arguments used by the upstream constructors.

The Super Learner is an ensemble:
  {VGG16, ResNet164, WideResNet28_10, MobileNet} -> pre-softmax logits
  -> stacked (N, 10, 4) -> 1x1 Conv1D meta-learner -> softmax (N, 10).
"""

import os
import sys
import types

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.join(_PROJECT_ROOT, "checkpoints", "external", "mnist_super_learner_repo")
_MODELS = f"{_REPO}/models"

_ensemble = None  # cached: (super_learner, [base_models])
_tf = None


def _patch_keras_for_legacy_repo():
    sys.modules.setdefault("utils", types.ModuleType("utils"))
    _train = types.ModuleType("train")
    _train.train = lambda *a, **kw: None
    sys.modules.setdefault("train", _train)

    from keras.layers import DepthwiseConv2D
    import keras.applications.mobilenet as km
    km.DepthwiseConv2D = DepthwiseConv2D

    from keras import optimizers as _opt
    for name in ("SGD", "RMSprop", "Adam"):
        cls = getattr(_opt, name)
        if getattr(cls, "_smc_sb_patched", False):
            continue
        orig_init = cls.__init__

        def make_new_init(orig):
            def new_init(self, *args, **kwargs):
                if "lr" in kwargs:
                    kwargs["learning_rate"] = kwargs.pop("lr")
                kwargs.pop("decay", None)
                orig(self, *args, **kwargs)
            return new_init

        cls.__init__ = make_new_init(orig_init)
        cls._smc_sb_patched = True


def _build_ensemble():
    global _ensemble, _tf
    if _ensemble is not None:
        return _ensemble

    _patch_keras_for_legacy_repo()
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    import tensorflow as tf
    _tf = tf

    from vgg16 import VGG16
    from resnet164 import ResNet164
    from wide_resnet_28_10 import WideResNet28_10
    from mobilenet import MobileNet
    from super_learner import SuperLearner

    with tf.device("/CPU:0"):
        bases = [VGG16(), ResNet164(), WideResNet28_10(), MobileNet()]
        weight_files = ["VGG16.h5", "ResNet164.h5", "WideResNet28_10.h5", "MobileNet.h5"]
        for m, wf in zip(bases, weight_files):
            m.load_weights(f"{_MODELS}/{wf}")

        super_learner = SuperLearner(bases)
        super_learner.load_weights(f"{_MODELS}/SuperLearner.h5")

    _ensemble = (super_learner, bases)
    return _ensemble


def load_mnist_super_learner(device):
    super_learner, _bases = _build_ensemble()
    tf = _tf
    from smc_sb_image import adapt_image_batch_to_shape

    @torch.no_grad()
    def classify(x_img: torch.Tensor) -> torch.Tensor:
        x = adapt_image_batch_to_shape(x_img, (1, 28, 28), pad_value=-1.0)
        x = x.clamp(-1, 1).detach().cpu()
        x = (x + 1) / 2
        x_np = x.permute(0, 2, 3, 1).numpy().astype(np.float32)  # (N, 28, 28, 1)
        with tf.device("/CPU:0"):
            scores = super_learner.get_scores(x_np)       # (N, 10, 4)
            probs = super_learner.model.predict(scores, verbose=0)  # (N, 10)
        return torch.from_numpy(np.asarray(probs)).to(device)

    return classify
