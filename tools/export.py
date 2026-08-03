"""PyTorch -> .ngm exporter.

Traces a small handful of module types rather than trying to be general: this
engine targets edge-sized CNNs, and a narrow, obviously-correct exporter beats
a general one that silently mis-folds something.

BatchNorm is folded into the preceding convolution here, at export time, so the
runtime never has to know BatchNorm exists.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Builder:
    def __init__(self, name: str):
        self.name = name
        self.tensors: list[dict] = []
        self.nodes: list[str] = []
        self.blob = bytearray()
        self.input_id = -1
        self.output_id = -1

    def add_activation(self, name: str, shape, dtype="f32", scale=0.0, zp=0) -> int:
        self.tensors.append({"name": name, "shape": list(shape), "dtype": dtype,
                             "is_weight": 0, "scale": scale, "zp": zp, "offset": 0})
        return len(self.tensors) - 1

    def add_weight(self, name: str, array: np.ndarray, dtype="f32",
                   scales: np.ndarray | None = None, zp=0) -> int:
        offset = len(self.blob)
        if dtype == "f32":
            self.blob += array.astype("<f4").tobytes()
            scale = 0.0
        else:
            self.blob += array.astype(np.int8).tobytes()
            # per-output-channel scales stored immediately after the weights
            self.blob += np.asarray(scales, dtype="<f4").tobytes()
            scale = float(np.mean(scales))
        self.tensors.append({"name": name, "shape": list(array.shape), "dtype": dtype,
                             "is_weight": 1, "scale": scale, "zp": zp, "offset": offset})
        return len(self.tensors) - 1

    def node(self, op: str, inputs: list[int], outputs: list[int], **attrs):
        parts = [f"N {op} {len(inputs)} " + " ".join(map(str, inputs)),
                 f"{len(outputs)} " + " ".join(map(str, outputs))]
        for k, v in attrs.items():
            parts.append(f"{k}={int(v) if isinstance(v, bool) else v}")
        self.nodes.append(" ".join(parts))

    def write(self, path: Path):
        lines = ["NGM1", f"name {self.name}", f"weight_bytes {len(self.blob)}",
                 f"tensors {len(self.tensors)}"]
        for t in self.tensors:
            lines.append(
                f"T {t['name']} {t['dtype']} {t['is_weight']} {t['scale']:.8g} "
                f"{t['zp']} {t['offset']} {len(t['shape'])} "
                + " ".join(map(str, t["shape"]))
            )
        lines.append(f"nodes {len(self.nodes)}")
        lines.extend(self.nodes)
        lines.append(f"io {self.input_id} {self.output_id}")
        lines.append("DATA")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(("\n".join(lines) + "\n").encode())
            f.write(bytes(self.blob))
        print(f"wrote {path} ({len(self.blob) / 1024:.1f} KB weights, "
              f"{len(self.nodes)} nodes)")


def _pair(v) -> tuple[int, int]:
    """PyTorch accepts an int or a 2-tuple for stride/padding/kernel; the engine
    always wants both axes spelled out. Collapsing them to one number silently
    breaks non-square kernels."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold BatchNorm into conv weights. Standard, but worth being explicit:
    w' = w * gamma/sqrt(var+eps), b' = (b - mean) * gamma/sqrt(var+eps) + beta."""
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma / torch.sqrt(var + bn.eps)
    w = w * scale.reshape(-1, 1, 1, 1)
    b = (b - mean) * scale + beta
    return w.numpy(), b.numpy()


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8 weights."""
    flat = w.reshape(w.shape[0], -1)
    amax = np.maximum(np.abs(flat).max(axis=1), 1e-12)
    scales = amax / 127.0
    q = np.round(flat / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(w.shape), scales.astype(np.float32)


def export(model: nn.Module, example: torch.Tensor, out: Path, name: str,
           quantize: bool = False, calib: torch.Tensor | None = None):
    """Walk a Sequential-style model and emit nodes.

    Activation ranges for the quantized path come from a calibration pass, so
    the exporter runs the model to collect them rather than guessing.
    """
    model = model.eval()
    layers = [m for m in model.modules()
              if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
                                nn.Linear, nn.Flatten, nn.AdaptiveAvgPool2d, nn.Dropout))]

    # Activation ranges for the int8 path, keyed by the module that produced
    # them. An earlier version tracked a running index into a list, which drifted
    # out of step wherever layers were fused or skipped and silently assigned the
    # wrong scale to a tensor.
    ranges: dict[int, float] = {}
    with torch.no_grad():
        cur = calib if calib is not None else example
        input_range = float(cur.abs().max())
        for m in layers:
            cur = m(cur)
            ranges[id(m)] = float(cur.abs().max())

    def scale_after(mods) -> float:
        """Output scale for a fused group: the range after its last module."""
        for m in reversed(mods):
            if id(m) in ranges:
                return ranges[id(m)] / 127.0
        return input_range / 127.0

    b = Builder(name)
    shape = list(example.shape)
    act_scale = input_range / 127.0 if quantize else 0.0
    cur_id = b.add_activation("input", shape, "f32")
    b.input_id = cur_id

    if quantize:
        qid = b.add_activation("input_q", shape, "i8", act_scale, 0)
        b.node("quantize", [cur_id], [qid])
        cur_id = qid

    i = 0
    while i < len(layers):
        m = layers[i]
        nxt = layers[i + 1] if i + 1 < len(layers) else None

        if isinstance(m, nn.Conv2d):
            if isinstance(nxt, nn.BatchNorm2d):
                w, bias = fold_bn(m, nxt)
                consumed = 2
            else:
                w = m.weight.detach().numpy()
                bias = (m.bias.detach().numpy() if m.bias is not None
                        else np.zeros(w.shape[0], dtype=np.float32))
                consumed = 1
            fuse_relu = isinstance(layers[i + consumed] if i + consumed < len(layers) else None,
                                   nn.ReLU)
            if fuse_relu:
                consumed += 1

            depthwise = m.groups == w.shape[0] and w.shape[1] == 1
            if quantize:
                qw, scales = quantize_per_channel(w)
                wid = b.add_weight(f"w{i}", qw, "i8", scales)
            else:
                wid = b.add_weight(f"w{i}", w.astype(np.float32))
            bid = b.add_weight(f"b{i}", bias.astype(np.float32))

            with torch.no_grad():
                probe = torch.zeros(shape)
                oshape = list(m(probe).shape)
            shape = oshape
            out_scale = scale_after(layers[i:i + consumed]) if quantize else 0.0
            oid = b.add_activation(f"a{i}", shape, "i8" if quantize else "f32",
                                   out_scale, 0)
            sh, sw = _pair(m.stride)
            ph, pw = _pair(m.padding)
            b.node("dwconv2d" if depthwise else "conv2d", [cur_id, wid, bid], [oid],
                   stride_h=sh, stride_w=sw, pad_h=ph, pad_w=pw, groups=m.groups,
                   relu=1 if fuse_relu else 0)
            cur_id = oid
            i += consumed
            continue

        if isinstance(m, nn.MaxPool2d):
            with torch.no_grad():
                shape = list(m(torch.zeros(shape)).shape)
            oid = b.add_activation(f"a{i}", shape,
                                   "i8" if quantize else "f32",
                                   b.tensors[cur_id]["scale"], 0)
            b.node("maxpool2d", [cur_id], [oid], k=_pair(m.kernel_size)[0],
                   pool_stride=_pair(m.stride)[0], pool_pad=_pair(m.padding)[0])
            cur_id = oid
            i += 1
            continue

        if isinstance(m, nn.AdaptiveAvgPool2d):
            shape = [shape[0], shape[1], 1, 1]
            # stay in int8 through pooling so the following Linear still gets the
            # int8 input its quantized weights require
            oid = b.add_activation(f"a{i}", shape, "i8" if quantize else "f32",
                                   scale_after([m]) if quantize else 0.0, 0)
            b.node("gap", [cur_id], [oid])
            cur_id = oid
            i += 1
            continue

        if isinstance(m, nn.Flatten):
            shape = [shape[0], int(np.prod(shape[1:]))]
            oid = b.add_activation(f"a{i}", shape, b.tensors[cur_id]["dtype"],
                                   b.tensors[cur_id]["scale"], 0)
            b.node("flatten", [cur_id], [oid])
            cur_id = oid
            i += 1
            continue

        if isinstance(m, nn.Linear):
            w = m.weight.detach().numpy()
            bias = (m.bias.detach().numpy() if m.bias is not None
                    else np.zeros(w.shape[0], dtype=np.float32))
            fuse_relu = isinstance(nxt, nn.ReLU)
            last = not any(isinstance(k, nn.Linear) for k in layers[i + 1:])
            if quantize:
                qw, scales = quantize_per_channel(w)
                wid = b.add_weight(f"w{i}", qw, "i8", scales)
            else:
                wid = b.add_weight(f"w{i}", w.astype(np.float32))
            bid = b.add_weight(f"b{i}", bias.astype(np.float32))
            shape = [shape[0], w.shape[0]]
            # the final layer produces float logits even in the quantized graph,
            # so the caller never has to dequantize
            odtype = "f32" if (last or not quantize) else "i8"
            group = [m, nxt] if fuse_relu else [m]
            oscale = 0.0 if odtype == "f32" else scale_after(group)
            oid = b.add_activation(f"a{i}", shape, odtype, oscale, 0)
            b.node("linear", [cur_id, wid, bid], [oid], relu=1 if fuse_relu else 0)
            cur_id = oid
            i += 2 if fuse_relu else 1
            continue

        if isinstance(m, nn.ReLU):
            oid = b.add_activation(f"a{i}", shape, b.tensors[cur_id]["dtype"],
                                   b.tensors[cur_id]["scale"], 0)
            b.node("relu", [cur_id], [oid])
            cur_id = oid
            i += 1
            continue

        if isinstance(m, nn.Dropout):
            i += 1
            continue

        raise TypeError(f"exporter does not handle {type(m).__name__}")

    if quantize and b.tensors[cur_id]["dtype"] == "i8":
        oid = b.add_activation("output_f32", shape, "f32")
        b.node("dequantize", [cur_id], [oid])
        cur_id = oid

    b.output_id = cur_id
    b.write(out)
    return b


def _pack_input(x: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(x.detach().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="mnist_cnn")
    args = ap.parse_args()
    from tools.models import build

    model, example, name = build(args.which)
    export(model, example, Path(f"models/{name}.ngm"), name)
    _pack_input(example, Path(f"models/{name}_input.bin"))
    struct.pack("f", 0.0)
