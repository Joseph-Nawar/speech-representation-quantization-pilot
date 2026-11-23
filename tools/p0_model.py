"""Read tensor shapes from the pinned PyTorch ZIP prefix without unpickling it."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from pickletools import genops
import re
import struct
from tempfile import NamedTemporaryFile

PROJECTIONS = {
    "attention.q_proj.weight": (768, 768),
    "attention.k_proj.weight": (768, 768),
    "attention.v_proj.weight": (768, 768),
    "attention.out_proj.weight": (768, 768),
    "feed_forward.intermediate_dense.weight": (3072, 768),
    "feed_forward.output_dense.weight": (768, 3072),
}
POSITION_GATE = "attention.gru_rel_pos_linear.weight"
PARAMETER = re.compile(r"(?:masked_spec_embed|(?:encoder|feature_extractor|feature_projection)\.[\w.]+)")
INTEGER_OPS = {"BININT", "BININT1", "BININT2"}


def digest(data):
    return sha256(data).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(obj, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def inspect_prefix(prefix):
    fields = struct.unpack_from("<4s5H3L2H", prefix)
    name_length, extra_length = fields[-2:]
    payload_start = 30 + name_length + extra_length
    if fields[0] != b"PK\x03\x04" or fields[3] != 0:
        raise ValueError("Expected an uncompressed local ZIP entry")
    if prefix[30:30 + name_length] != b"archive/data.pkl":
        raise ValueError("Unexpected checkpoint metadata entry")
    ops = [(operation.name, argument, position) for operation, argument, position in genops(prefix[payload_start:])]
    globals_seen = {arg for name, arg, _ in ops if name == "GLOBAL"}
    if globals_seen != {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage"}:
        raise ValueError("Unexpected serialization or checkpoint dtype")

    shapes = {}
    name = None
    for index, (opcode, argument, _) in enumerate(ops):
        if opcode == "BINUNICODE" and PARAMETER.fullmatch(argument):
            name = argument
        if opcode == "BINPERSID":
            if name is None or name in shapes or ops[index + 1][:2] != ("BININT1", 0):
                raise ValueError("Missing, duplicated, or offset tensor")
            dimensions = []
            shape_start = index + 2
            if ops[shape_start][0] == "MARK":
                shape_start += 1
            for next_op, value, _ in ops[shape_start:]:
                if next_op in {"TUPLE1", "TUPLE2", "TUPLE3", "TUPLE"}:
                    break
                if next_op not in INTEGER_OPS or value <= 0:
                    raise ValueError(f"Unexpected shape encoding for {name}")
                dimensions.append(value)
            if not dimensions:
                raise ValueError(f"Missing dimensions for {name}")
            shapes[name] = tuple(dimensions)
            name = None
    return shapes, prefix[payload_start:payload_start + ops[-1][2] + 1]


def cost(shape, bits):
    rows, columns = shape
    groups, remainder = divmod(columns, 64)
    tail = ((remainder * bits + 7) // 8 + 2) if remainder else 0
    return rows * (groups * (8 * bits + 2) + tail)


def inventory(shapes):
    listed = {f"encoder.layers.{index}.{name}": shape for index in range(12) for name, shape in PROJECTIONS.items()}
    gates = {f"encoder.layers.{index}.{POSITION_GATE}": (8, 64) for index in range(12)}
    observed_blocks = Counter(re.match(r"encoder\.layers\.(\d+)\.", name)[1] for name in shapes if name.startswith("encoder.layers."))
    if set(observed_blocks) != {str(index) for index in range(12)}:
        raise ValueError("Unexpected transformer block indices")
    for name, shape in (listed | gates).items():
        if shapes.get(name) != shape:
            raise ValueError(f"Missing or unexpected shape for {name}")

    total = sum(map(prod, shapes.values()))
    interpretations = {}
    for interpretation, eligible in {"six_named": listed, "seven_linear_including_gate": listed | gates}.items():
        eligible_elements = sum(map(prod, eligible.values()))
        excluded = total - eligible_elements
        interpretations[interpretation] = {
            "eligible_matrices": len(eligible),
            "eligible_elements": eligible_elements,
            "excluded_elements": excluded,
            "by_baseline_dtype": {
                dtype: {
                    "eligible_baseline_bytes": eligible_elements * width,
                    "excluded_baseline_bytes": excluded * width,
                    **{f"uniform{bits}_calculated_total_bytes": excluded * width + sum(cost(shape, bits) for shape in eligible.values()) for bits in (4, 8)}
                }
                for dtype, width in (("fp16", 2), ("fp32", 4))
            },
        }
    return {
        "status": "BLOCKED",
        "block_indices": list(range(12)),
        "serialized_tensor_count": len(shapes),
        "serialized_float32_elements": total,
        "named_projection_shapes": PROJECTIONS,
        "additional_linear": {POSITION_GATE: (8, 64)},
        "interpretations": interpretations,
        "tensor_shapes": shapes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.read_bytes()
    shapes, metadata = inspect_prefix(prefix)
    result = inventory(shapes)
    result.update(checkpoint_prefix_sha256=digest(prefix), checkpoint_tensor_metadata_sha256=digest(metadata))
    write_json(args.out, result)
    print(f"BLOCKED: {len(shapes)} tensors, 12 transformer blocks, two distinct eligibility readings")


if __name__ == "__main__":
    main()
