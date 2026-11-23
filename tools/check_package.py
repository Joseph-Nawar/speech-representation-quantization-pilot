"""Dependency-free handoff sanity check (does NOT validate research outcomes)."""
from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'README.md', 'AGENTS.md', 'configs/pilot.yaml',
    'prompts/CODEX_START.md', 'prompts/CODEX_AFTER_PREFLIGHT.md',
    'docs/01_protocol.md', 'docs/02_data_and_model.md',
    'docs/03_quantization_and_baselines.md', 'docs/04_analysis_and_gates.md',
    'docs/05_literature_and_novelty.md', 'docs/06_execution_and_artifacts.md',
    'docs/07_report_templates.md', 'schemas/decision.json', '.gitignore',
]
for filename in REQUIRED:
    path = ROOT / filename
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f'MISSING/EMPTY: {filename}')
    print(f'{sha256(path.read_bytes()).hexdigest()}  {filename}')
print(f'OK: {len(REQUIRED)} required files present.')
