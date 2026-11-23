from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
import re

import torch

from tools.p0_model import POSITION_GATE, PROJECTIONS

GROUP = 64
BITS = (4, 8)
BLOCKS = 12
ROLES = PROJECTIONS | {POSITION_GATE: (8, 64)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packed_bytes(rows, columns, bits):
    require(bits in BITS and rows > 0 and columns > 0, "Invalid packed matrix geometry")
    full, remainder = divmod(columns, GROUP)
    tail = (remainder * bits + 7) // 8 + 2 if remainder else 0
    return rows * (full * (GROUP * bits // 8 + 2) + tail)


def encode(weight, bits):
    """Round ties to even using the stored FP16 scale and reconstruct in reference dtype."""
    require(bits in BITS and weight.ndim == 2 and min(weight.shape) > 0, "Invalid quantization input")
    rows, columns = weight.shape
    values = weight.detach().to(device="cpu", dtype=torch.float32)
    require(torch.isfinite(values).all().item(), "Non-finite quantization input")
    padded = torch.nn.functional.pad(values, (0, (-columns) % GROUP)).reshape(rows, -1, GROUP)
    qmax = (1 << (bits - 1)) - 1
    peak = padded.abs().amax(dim=-1)
    exact = peak / qmax
    stored = exact.half()
    require(torch.isfinite(stored).all().item() and not ((peak != 0) & (stored == 0)).any().item(),
            "Nonzero group scale cannot be represented as FP16")
    denominator = torch.where(stored == 0, 1, stored).float().unsqueeze(-1)
    codes = (padded / denominator).round().clamp(-qmax, qmax).to(torch.int8)
    reconstructed = (codes.float() * stored.float().unsqueeze(-1)).reshape(rows, -1)[:, :columns]
    reconstructed = reconstructed.to(dtype=weight.dtype)
    require(torch.isfinite(reconstructed).all().item(), "Reconstruction overflow")
    return codes.reshape(rows, -1)[:, :columns], stored, reconstructed


def tensor_hash(value):
    contiguous = value.detach().cpu().contiguous()
    digest = sha256(str((str(contiguous.dtype), tuple(contiguous.shape))).encode())
    digest.update(memoryview(contiguous.numpy()).cast("B"))
    return digest.hexdigest()


def fingerprint(model):
    return {name: tensor_hash(weight) for name, weight in model.named_parameters()}


def master_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def stable_hash(value):
    contents = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(contents.encode()).hexdigest()


def discover(model, config, expected_serialized):
    """Verify the complete checkpoint and resolve the exact allowlist on runtime modules."""
    quant = config["quantization"]
    require(set(quant["eligible"]) == PROJECTIONS.keys() and len(quant["eligible"]) == len(PROJECTIONS)
            and quant["excluded_block_linear"] == POSITION_GATE and quant["group_size"] == GROUP
            and tuple(quant["bits"]) == BITS and quant["scale_dtype"] == "float16",
            "Config disagrees with Q-001")
    require(len(model.encoder.layers) == BLOCKS == config["model"]["expected_transformer_blocks"], "Wrong block count")
    parameters = dict(model.named_parameters())
    # Torch parametrizes the original positional convolution's two serialized tensors.
    aliases = {"encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
               "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1"}
    expected_parameters = {aliases.get(name, name): tuple(shape) for name, shape in expected_serialized.items()}
    require({name: tuple(value.shape) for name,value in parameters.items()} == expected_parameters,
            "Runtime parameter inventory differs from the complete pinned checkpoint")

    in_block = {
        f"{name}.weight": module.weight
        for name,module in model.named_modules()
        if re.match(r"^encoder\.layers\.\d+\.", name) and isinstance(module, torch.nn.Linear)
    }
    expected = {f"encoder.layers.{i}.{role}": shape for i in range(BLOCKS) for role,shape in ROLES.items()}
    require({name: tuple(value.shape) for name,value in in_block.items()} == expected,
            "Expected exactly seven named linears in every pinned WavLM block")

    rows = []
    selection = []
    for i in range(BLOCKS):
        names = []
        for role in ROLES:
            name = f"encoder.layers.{i}.{role}"
            value = in_block[name]
            require(parameters[name] is value, f"Unregistered linear {name}")
            eligible = role in PROJECTIONS
            rows.append(dict(block=i+1,path=name,shape=list(value.shape),dtype=str(value.dtype),
                             parameters=value.numel(),groups=value.shape[0] * ((value.shape[1]+GROUP-1)//GROUP),
                             selected=eligible,
                             calculated_packed_bytes={str(bit): packed_bytes(*value.shape, bit) for bit in BITS}
                             if eligible else None))
            if eligible:
                names.append(name)
        selection.append(tuple(names))
    return tuple(selection), rows


@dataclass(frozen=True)
class CostBook:
    fixed: int
    blocks: tuple[tuple[int, int], ...]

    def total(self, bitmap):
        require(len(bitmap) == len(self.blocks) and all(b in BITS for b in bitmap), "Invalid bit mapping")
        return self.fixed + sum(costs[BITS.index(bit)] for costs,bit in zip(self.blocks,bitmap,strict=True))

    def budget(self, fraction):
        f = Fraction(str(fraction))
        require(0 <= f <= 1, "Invalid budget fraction")
        lower = self.total((4,) * len(self.blocks))
        upper = self.total((8,) * len(self.blocks))
        return lower + f * (upper-lower)

    def feasible(self, fraction):
        ceiling = self.budget(fraction)
        return tuple(bitmap for bitmap in product(BITS, repeat=len(self.blocks)) if self.total(bitmap) <= ceiling)


def cost_book(model, selection, reference_dtype, probe_bytes=0):
    weights = dict(model.named_parameters())
    selected = {name for names in selection for name in names}
    width = torch.empty((), dtype=reference_dtype).element_size()
    fixed = probe_bytes + width * sum(value.numel() for name,value in weights.items() if name not in selected)
    blocks = tuple(tuple(sum(packed_bytes(*weights[name].shape, bits) for name in names) for bits in BITS)
                   for names in selection)
    return CostBook(fixed, blocks)


@contextmanager
def apply_from_master(model, master, names, bits):
    if not names or len(names) != len(set(names)):
        raise ValueError("Invalid selected weight names")
    parameters = dict(model.named_parameters())
    stats = {}
    modified = []
    try:
        for name in names:
            source = master[name]
            target = parameters[name]
            require(target.dtype == source.dtype and torch.equal(target.detach().cpu(), source),
                    f"Model is not restored for {name}")
        for name in names:
            source = master[name]
            _, _, restored = encode(source, bits)
            delta = restored.float() - source.float()
            stats[name] = {"changed": int(torch.count_nonzero(delta)), "max_abs_error": delta.abs().max().item(),
                           "rmse": delta.square().mean().sqrt().item()}
            modified.append(name)
            with torch.no_grad():
                parameters[name].copy_(restored)
        yield stats
    finally:
        with torch.no_grad():
            for name in modified:
                parameters[name].copy_(master[name])
