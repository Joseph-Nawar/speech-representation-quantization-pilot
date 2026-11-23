"""Probe the local runtime and inspect zero-storage WavLM module shapes."""
import argparse
import importlib.metadata
import platform
from pathlib import Path
import resource
import shutil
import subprocess
import time

from p0_model import write_json


def sysctl(name):
    return subprocess.check_output(["sysctl", "-n", name], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="cache/p0-model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    import torch
    from transformers import WavLMConfig, WavLMModel

    versions = {}
    for package in (
        "torch", "torchaudio", "soundfile", "librosa", "transformers", "pyarrow",
        "scikit-learn", "numpy", "scipy", "pytest", "PyYAML", "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    import soundfile
    import torchaudio
    mps_available = torch.backends.mps.is_available()
    operators = {}
    for device in ("cpu", "mps") if mps_available else ("cpu",):
        for precision in ("float16", "float32"):
            value = torch.ones((3, 3), device=device, dtype=getattr(torch, precision))
            first = (value @ value).cpu()
            if not torch.equal(first, (value @ value).cpu()) or not (first == 3).all():
                raise ValueError(f"Incorrect matrix multiply on {device} at {precision}")
            operators[f"{device}/{precision}"] = "PASS"

    cfg = WavLMConfig.from_pretrained(args.model_dir, local_files_only=True)
    with torch.device("meta"):
        model = WavLMModel(cfg)
    linears = {
        f"{name}.weight": list(module.weight.shape)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith("encoder.layers.")
    }
    result = {
        "os": platform.platform(),
        "macos_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
        "arch": platform.machine(),
        "hardware_model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "cpu_cores": int(sysctl("hw.ncpu")),
        "ram_bytes": int(sysctl("hw.memsize")),
        "disk_free_bytes": shutil.disk_usage(".").free,
        "python": platform.python_version(),
        "packages": versions,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "cuda_available": torch.cuda.is_available(),
        "synthetic_ops": operators,
        "soundfile_libsndfile": soundfile.__libsndfile_version__,
        "torchaudio_import": torchaudio.__version__,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "wavlm_meta": {
            "transformer_blocks": len(model.encoder.layers),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "block_linears": linears,
        },
        "audio_clips_processed": 0,
        "model_forward_passes": 0,
        "wall_seconds": round(time.monotonic() - start, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "unified_memory_peak_bytes": None,
    }
    write_json(args.out, result)
    print(f"PASS: {operators}; {len(linears)} WavLM block linears on meta; {result['wall_seconds']} s")


if __name__ == "__main__":
    main()
