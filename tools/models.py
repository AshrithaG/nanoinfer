"""The models this engine targets: small enough to run on a microcontroller-class
budget, real enough that the numbers mean something.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mnist_cnn() -> nn.Module:
    """The correctness target. ~100k parameters."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def kws_dscnn() -> nn.Module:
    """Keyword spotting on a 49x10 MFCC frame, DS-CNN style: one dense stem then
    depthwise-separable blocks. This is the flagship benchmark -- depthwise
    convolution is where a general runtime's dispatch overhead shows up most."""
    def block(c):
        return nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv2d(1, 64, (10, 4), stride=(2, 2), padding=(5, 1)),
        nn.BatchNorm2d(64), nn.ReLU(),
        block(64), block(64), block(64), block(64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 12),
    )


def vww_mobilenet() -> nn.Module:
    """Visual wake words at 96x96: the stretch target, where the memory planner
    and depthwise path both have to work to fit a realistic budget."""
    def dw_sep(cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin),
            nn.BatchNorm2d(cin), nn.ReLU(),
            nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        dw_sep(16, 32, 1),
        dw_sep(32, 64, 2),
        dw_sep(64, 64, 1),
        dw_sep(64, 128, 2),
        dw_sep(128, 128, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 2),
    )


REGISTRY = {
    "mnist_cnn": (mnist_cnn, (1, 1, 28, 28)),
    "kws_dscnn": (kws_dscnn, (1, 1, 49, 10)),
    "vww_mobilenet": (vww_mobilenet, (1, 3, 96, 96)),
}


def build(which: str, seed: int = 0):
    if which not in REGISTRY:
        raise KeyError(f"unknown model {which}; have {sorted(REGISTRY)}")
    fn, shape = REGISTRY[which]
    torch.manual_seed(seed)
    model = fn().eval()
    # BatchNorm running stats start at (0, 1), which is not representative;
    # push some data through so folding is exercised on realistic statistics
    model.train()
    with torch.no_grad():
        for _ in range(8):
            model(torch.randn(8, *shape[1:]))
    model.eval()
    example = torch.randn(*shape)
    return model, example, which
