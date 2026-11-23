"""P2-only official VCTK audio audit and sealed-test-safe split helpers."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import io
import json
from pathlib import Path
import tempfile
import os
import re
import zlib

import numpy as np
import soundfile as sf

from sqpilot.data import balanced_split, preprocess, unpack_member
from sqpilot.quant import require
from tools.p0_source_ranges import OFFICIAL, Remote, norm, write_bytes


MIN_SPEAKERS, MAX_SPEAKERS = 40, 64
MIN_PROMPTS, MAX_PROMPTS = 16, 24
MAX_CELLS = 1134
MAX_AUDIO_PAYLOAD_BYTES = 2 * 1024**3
MAX_LOCAL_DECODED_BYTES = 4 * 1024**3
MIN_FREE_DISK_BYTES = 5 * 1024**3
MAX_SINGLE_RANGE_BYTES = 18_000_000
SPLIT_SEED = 20261122
SPLIT_MINIMA = {
    "train": (8, 12),
    "dev": (3, 5),
    "test": (3, 5),
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def verify_manifest_hash(document: dict, expected_sha256: str) -> bool:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256.lower()):
        raise ValueError("Expected manifest SHA256 must be 64 hexadecimal characters")
    body = {key: value for key, value in document.items() if key != "sha256"}
    actual = sha256(canonical_json(body)).hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(f"Manifest SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return True


def preflight_resource_limits(
    resources: dict, *, cells: int, range_bytes: int, local_bytes: int, free_disk_bytes: int
) -> dict:
    max_cells = int(resources["P2_max_audio_cells"])
    max_bytes = int(resources["P2_max_audio_payload_bytes"])
    max_local = int(resources["P2_max_local_decoded_bytes"])
    min_free = int(resources["P2_min_free_disk_bytes"])
    if cells <= 0 or cells > max_cells:
        raise ValueError(f"RESOURCE_OVERRIDE_REQUIRED: cells {cells} exceed authorized cap {max_cells}")
    if range_bytes < 0 or range_bytes > max_bytes:
        raise ValueError(f"RESOURCE_OVERRIDE_REQUIRED: range bytes {range_bytes} exceed cap {max_bytes}")
    if local_bytes < 0 or local_bytes > max_local:
        raise ValueError(f"RESOURCE_OVERRIDE_REQUIRED: local decoded/extraction bytes {local_bytes} exceed cap {max_local}")
    if free_disk_bytes < min_free:
        raise ValueError(f"RESOURCE_OVERRIDE_REQUIRED: free disk {free_disk_bytes} is below minimum {min_free}")
    return {
        "planned_cells": cells,
        "planned_range_bytes": range_bytes,
        "max_audio_cells": max_cells,
        "max_audio_payload_bytes": max_bytes,
        "planned_local_decoded_and_extracted_bytes": local_bytes,
        "max_local_decoded_bytes": max_local,
        "free_disk_bytes_before_download": free_disk_bytes,
        "min_free_disk_bytes": min_free,
    }


def plan_local_upper_bound(
    plan: dict,
    *,
    max_duration_seconds: float,
    sample_rate_hz: int,
    bytes_per_sample: int = 4,
) -> int:
    """Conservatively bound extracted FLAC bytes plus all full-length PCM samples."""
    cells = int(plan.get("candidate_cells", 0))
    flac_bytes = int(plan.get("planned_flac_bytes", -1))
    if cells <= 0 or flac_bytes < 0 or max_duration_seconds <= 0 or sample_rate_hz <= 0 or bytes_per_sample <= 0:
        raise ValueError("invalid local audio upper-bound inputs")
    pcm_bytes = cells * int(np.ceil(max_duration_seconds * sample_rate_hz)) * bytes_per_sample
    return flac_bytes + pcm_bytes


def write_json_atomic(path: Path, document: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(document) + b"\n"
    digest = sha256(payload).hexdigest()
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise ValueError(f"Refusing to overwrite changed stage artifact {path}")
        return digest
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resumable download record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"resumable download record must be an object: {path}")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _attempt_start_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("attempt-*.json")
        if len(path.name) == 17 and path.name.startswith("attempt-") and path.name.endswith(".json")
        and path.name[8:12].isdigit()
    )


def materialize_official(
    plan: dict,
    audio_dir: Path,
    receipt_dir: Path,
    *,
    max_payload_bytes: int = MAX_AUDIO_PAYLOAD_BYTES,
    batch_size: int = 16,
) -> dict:
    """Resume verified single-member HTTP ranges under a cumulative byte cap."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = sorted(plan.get("rows", []), key=lambda value: value["id"])
    if not rows or len(rows) > MAX_CELLS or batch_size <= 0:
        raise ValueError("authorized audio row count or batch size is invalid")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip IDs in bounded audio plan")
    plan_hash = sha256(canonical_json(plan)).hexdigest()
    planned = sum(int(row["span"]) for row in rows)
    if planned > max_payload_bytes:
        raise ValueError(f"RESOURCE_OVERRIDE_REQUIRED: range payload exceeds cap: {planned} > {max_payload_bytes}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    attempts_root = receipt_dir / "attempts"
    pending = []
    cached = []
    for row in rows:
        clip_id = row["id"]
        if Path(clip_id).name != clip_id or not clip_id.endswith("_mic1"):
            raise ValueError(f"invalid anonymized mic1 clip ID {clip_id!r}")
        audio_path = audio_dir / f"{clip_id}.flac"
        receipt_path = receipt_dir / f"{clip_id}.json"
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            if receipt.get("plan_sha256") != plan_hash:
                raise ValueError("changed plan manifest invalidates resumable audio receipt")
            if not audio_path.is_file():
                raise ValueError(f"download receipt exists but audio is missing: {clip_id}")
            payload = audio_path.read_bytes()
            if (
                len(payload) != row["size"]
                or zlib.crc32(payload) != row["crc32"]
                or sha256(payload).hexdigest() != receipt.get("audio_sha256")
            ):
                raise ValueError(f"cached official audio failed receipt/CRC validation: {clip_id}")
            cached.append(row)
        else:
            pending.append(row)

    starts = []
    for path in (
        sorted(candidate for folder in attempts_root.iterdir() for candidate in _attempt_start_files(folder))
        if attempts_root.exists() else []
    ):
        attempt = _read_json(path)
        if attempt.get("plan_sha256") != plan_hash:
            raise ValueError("changed plan manifest invalidates previous range attempts")
        starts.append(attempt)
    reserved_before = sum(int(attempt["reserved_range_bytes"]) for attempt in starts)
    reserved_new = 0
    actual_this_call = 0
    extracted_bytes = 0
    audio_payload_bytes = 0
    completed = list(cached)

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        batch_reserved = sum(int(row["span"]) for row in batch)
        if reserved_before + reserved_new + batch_reserved > max_payload_bytes:
            raise ValueError("RESOURCE_OVERRIDE_REQUIRED: resumable range payload would exceed cumulative audio transfer cap")
        attempt_numbers = {}
        for row in batch:
            attempt_folder = attempts_root / row["id"]
            attempt_folder.mkdir(parents=True, exist_ok=True)
            attempt_number = 1 + len(_attempt_start_files(attempt_folder))
            attempt_numbers[row["id"]] = attempt_number
            attempt_path = attempt_folder / f"attempt-{attempt_number:04d}.json"
            write_json_atomic(
                attempt_path,
                {
                    "id": row["id"],
                    "plan_sha256": plan_hash,
                    "reserved_range_bytes": int(row["span"]),
                    "status": "STARTED",
                },
            )
            reserved_new += int(row["span"])

        def fetch(row):
            source = Remote(OFFICIAL, 11747302977, '"8a6ba2946b36fcbef0212cad601f4bfa"')
            archive_range = source.pull(int(row["offset"]), int(row["span"]))
            payload = unpack_member(archive_range, row)
            return row, payload, source.bytes

        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            futures = [executor.submit(fetch, row) for row in batch]
            for future in as_completed(futures):
                row, payload, transfer = future.result()
                clip_id = row["id"]
                audio_path = audio_dir / f"{clip_id}.flac"
                if audio_path.exists():
                    if audio_path.read_bytes() != payload:
                        raise ValueError(f"unreceipted local audio differs from official member: {clip_id}")
                else:
                    _atomic_write_bytes(audio_path, payload)
                receipt = {
                    "id": clip_id,
                    "plan_sha256": plan_hash,
                    "official_path": row["path"],
                    "official_offset": int(row["offset"]),
                    "range_bytes": int(transfer),
                    "compressed_audio_payload_bytes": int(row["compressed"]),
                    "file_bytes": len(payload),
                    "audio_sha256": sha256(payload).hexdigest(),
                    "official_crc32": int(row["crc32"]),
                    "source": "official_vctk_0.92",
                    "microphone": "mic1",
                }
                write_json_atomic(receipt_dir / f"{clip_id}.json", receipt)
                result_path = attempts_root / clip_id / f"attempt-{attempt_numbers[clip_id]:04d}-result.json"
                write_json_atomic(
                    result_path,
                    {"id": clip_id, "status": "SUCCESS", "actual_range_bytes": int(transfer),
                     "audio_sha256": receipt["audio_sha256"], "plan_sha256": plan_hash},
                )
                actual_this_call += int(transfer)
                extracted_bytes += len(payload)
                audio_payload_bytes += int(row["compressed"])
                completed.append(row)
        print(
            f"P2 G0 range progress: {min(start + len(batch), len(pending))}/{len(pending)} newly requested; "
            f"{len(completed)}/{len(rows)} cells verified",
            flush=True,
        )

    # Include previously cached extracted bytes/audio payload in stage totals.
    for row in cached:
        payload = (audio_dir / f'{row["id"]}.flac').read_bytes()
        extracted_bytes += len(payload)
        audio_payload_bytes += int(row["compressed"])
    reserved_total = reserved_before + reserved_new
    if reserved_total > max_payload_bytes:
        raise ValueError("RESOURCE_OVERRIDE_REQUIRED: cumulative range payload exceeded configured cap")
    actual_cumulative = 0
    if attempts_root.exists():
        for folder in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
            for result_path in sorted(folder.glob("attempt-*-result.json")):
                attempt_result = _read_json(result_path)
                if attempt_result.get("plan_sha256") != plan_hash:
                    raise ValueError("changed plan manifest invalidates a prior range result")
                if attempt_result.get("status") != "SUCCESS":
                    raise ValueError("invalid completed range receipt status")
                actual_cumulative += int(attempt_result["actual_range_bytes"])
    return {
        "plan_sha256": plan_hash,
        "planned_range_bytes": planned,
        "reserved_range_bytes_total": reserved_total,
        "actual_range_bytes": actual_cumulative,
        "actual_range_bytes_this_call": actual_this_call,
        "compressed_audio_payload_bytes": audio_payload_bytes,
        "extracted_flac_bytes": extracted_bytes,
        "downloaded_cells": len(pending),
        "cached_cells": len(cached),
        "total_verified_cells": len(completed),
        "completed_ids": sorted(row["id"] for row in completed),
        "max_payload_bytes": max_payload_bytes,
    }


def select_max_complete_grid(
    cells,
    *,
    min_speakers: int = MIN_SPEAKERS,
    max_speakers: int = MAX_SPEAKERS,
    min_prompts: int = MIN_PROMPTS,
    max_prompts: int = MAX_PROMPTS,
    max_cells: int = MAX_CELLS,
) -> dict:
    """Maximize cells, speakers, prompts, then choose lexicographically first IDs."""
    values = list(cells)
    if len(values) != len(set(values)):
        raise ValueError("duplicate candidate cell")
    if not values or any(not isinstance(s, str) or not s or not isinstance(p, str) or not p for s, p in values):
        raise ValueError("candidate cells must contain non-empty string IDs")
    if min_speakers > max_speakers or min_prompts > max_prompts or max_cells <= 0:
        raise ValueError("invalid complete-grid bounds")
    available_speakers = sorted({s for s, _ in values})
    available_prompts = sorted({p for _, p in values})
    membership = set(values)
    best = None
    best_priority = None
    prompt_cap = min(max_prompts, len(available_prompts))
    for n_prompts in range(prompt_cap, min_prompts - 1, -1):
        for prompt_subset in combinations(available_prompts, n_prompts):
            speaker_subset = tuple(
                speaker for speaker in available_speakers
                if all((speaker, prompt) in membership for prompt in prompt_subset)
            )
            n_speakers = min(max_speakers, len(speaker_subset), max_cells // n_prompts)
            if n_speakers < min_speakers:
                continue
            selected_speakers = speaker_subset[:n_speakers]
            candidate_cells = tuple(product(selected_speakers, prompt_subset))
            priority = (len(candidate_cells), n_speakers, n_prompts)
            lexical_key = (selected_speakers, prompt_subset)
            if (
                best is None
                or priority > best_priority
                or (priority == best_priority and lexical_key < best["_lexical_key"])
            ):
                best_priority = priority
                best = {
                    "speakers": list(selected_speakers),
                    "prompts": list(prompt_subset),
                    "cells": [list(cell) for cell in candidate_cells],
                    "_lexical_key": lexical_key,
                }
    if best is None:
        raise ValueError(
            f"No complete grid meets {min_speakers}–{max_speakers} speakers × "
            f"{min_prompts}–{max_prompts} prompts under {max_cells} cells"
        )
    best.pop("_lexical_key")
    return best


def build_official_member_plan(
    candidate: dict,
    official_index: list,
    official_texts: dict,
    *,
    max_cells: int = MAX_CELLS,
    max_range_bytes: int = MAX_AUDIO_PAYLOAD_BYTES,
) -> dict:
    speakers = sorted(candidate.get("speakers", []))
    prompts = sorted(candidate.get("prompts", []))
    cells = list(product(speakers, prompts))
    if len(cells) > max_cells or not speakers or not prompts:
        raise ValueError("P0 candidate grid exceeds the authorized cell bound or is empty")
    if len(speakers) != len(set(speakers)) or len(prompts) != len(set(prompts)):
        raise ValueError("duplicate speaker or prompt ID in P0 candidate grid")
    index_by_path = {}
    offsets = []
    for member in official_index:
        if len(member) != 5:
            raise ValueError("official ZIP index row has an unexpected shape")
        path, crc, size, compressed, offset = member
        if path in index_by_path:
            raise ValueError(f"duplicate official ZIP path: {path}")
        index_by_path[path] = member
        offsets.append((int(offset), path))
    offsets.sort()
    if len({offset for offset, _ in offsets}) != len(offsets):
        raise ValueError("official ZIP member headers have duplicate offsets")
    next_offset = {path: offsets[i + 1][0] for i, (_, path) in enumerate(offsets[:-1])}

    canonical_prompts = {}
    rows = []
    for speaker, prompt in cells:
        if not re.fullmatch(r"(?:p\d+|s\d+)", speaker) or not re.fullmatch(r"\d{3}", prompt):
            raise ValueError("unexpected non-anonymized VCTK speaker or prompt ID")
        path = f"wav48_silence_trimmed/{speaker}/{speaker}_{prompt}_mic1.flac"
        if not path.endswith("_mic1.flac"):
            raise ValueError("candidate audio path is not mic1")
        if path not in index_by_path or path not in next_offset:
            raise ValueError(f"official mic1 member missing: {path}")
        normalized_text = norm(official_texts.get(f"{speaker}_{prompt}", ""))
        if not normalized_text:
            raise ValueError(f"official transcript missing or empty: {speaker}_{prompt}")
        if canonical_prompts.setdefault(prompt, normalized_text) != normalized_text:
            raise ValueError(f"inconsistent canonical prompt transcript: {prompt}")
        _, crc, size, compressed, offset = index_by_path[path]
        span = next_offset[path] - offset
        if min(int(size), int(compressed), int(span)) <= 0 or compressed > span:
            raise ValueError(f"invalid bounded official ZIP member metadata: {path}")
        rows.append(
            {
                "id": f"{speaker}_{prompt}_mic1",
                "speaker": speaker,
                "prompt": prompt,
                "mic": "mic1",
                "path": path,
                "crc32": int(crc),
                "size": int(size),
                "compressed": int(compressed),
                "offset": int(offset),
                "span": int(span),
                "transcript_sha256": sha256(normalized_text.encode()).hexdigest(),
            }
        )
    range_bytes = sum(row["span"] for row in rows)
    flac_bytes = sum(row["size"] for row in rows)
    if max(row["span"] for row in rows) > MAX_SINGLE_RANGE_BYTES:
        raise ValueError("official mic1 member range exceeds the source client's 18 MB request ceiling")
    if range_bytes > max_range_bytes:
        raise ValueError(f"planned official ranges {range_bytes} exceed audio transfer cap {max_range_bytes}")
    return {
        "candidate_speakers": speakers,
        "candidate_prompts": prompts,
        "candidate_cells": len(rows),
        "planned_range_bytes": range_bytes,
        "planned_flac_bytes": flac_bytes,
        "rows": sorted(rows, key=lambda row: row["id"]),
    }


def verify_candidate_metadata(
    candidate: dict,
    official_index: list,
    official_texts: dict,
    mirror_rows: list[dict],
) -> dict:
    """Cross-audit the pinned candidate's mic1 IDs and transcript text against both sources."""
    speakers = sorted(candidate.get("speakers", []))
    prompts = sorted(candidate.get("prompts", []))
    expected = {(speaker, prompt) for speaker, prompt in product(speakers, prompts)}
    if len(expected) != len(speakers) * len(prompts) or not expected:
        raise ValueError("candidate speaker/prompt grid is empty or duplicated")
    official_paths = {}
    for row in official_index:
        if len(row) != 5:
            raise ValueError("official ZIP index row has an unexpected shape")
        path = row[0]
        if path in official_paths:
            raise ValueError(f"duplicate official ZIP path: {path}")
        official_paths[path] = row

    mirror_by_key = {}
    for row in mirror_rows:
        recording_id = row.get("id")
        mic_id = row.get("mic_id")
        if not isinstance(recording_id, str) or mic_id not in {"mic1", "mic2"}:
            raise ValueError("mirror metadata has an invalid recording ID or microphone")
        key = (recording_id, mic_id)
        if key in mirror_by_key:
            raise ValueError(f"duplicate mirror recording: {recording_id}/{mic_id}")
        mirror_by_key[key] = row

    for speaker, prompt in sorted(expected):
        identifier = f"{speaker}_{prompt}"
        official_path = f"wav48_silence_trimmed/{speaker}/{identifier}_mic1.flac"
        mirror_path = f"{speaker}/{identifier}_mic1.flac"
        text = norm(official_texts.get(identifier, ""))
        if not text or official_path not in official_paths:
            raise ValueError(f"candidate cell lacks official mic1 audio or transcript: {identifier}")
        mirror = mirror_by_key.get((identifier, "mic1"))
        if mirror is None:
            raise ValueError(f"candidate cell lacks pinned mirror mic1 metadata: {identifier}")
        if mirror.get("speaker_id") != speaker or mirror.get("path") != mirror_path or mirror.get("mic_id") != "mic1":
            raise ValueError(f"candidate mirror path or microphone differs: {identifier}")
        if norm(str(mirror.get("text_original", ""))) != text:
            raise ValueError(f"candidate mirror transcript differs from official source: {identifier}")
        mirror_duration = mirror.get("duration_s")
        if (not isinstance(mirror_duration, (int, float)) or not np.isfinite(mirror_duration)
                or not 1.0 <= float(mirror_duration) <= 10.0):
            raise ValueError(f"candidate mirror duration is outside the P0 metadata screen: {identifier}")
        if identifier not in official_texts:
            raise ValueError(f"candidate transcript is absent from official source: {identifier}")
    return {
        "candidate_cells_cross_audited": len(expected),
        "candidate_speakers": len(speakers),
        "candidate_prompts": len(prompts),
        "mirror_mic1_candidate_rows": len(expected),
        "official_mic1_candidate_rows": len(expected),
        "mirror_rows_audited_for_duplicate_recording_ids": len(mirror_rows),
        "transcript_normalization": "NFKC, casefold, curly apostrophe to ASCII apostrophe, whitespace collapse",
        "audio_parity": "UNVERIFIED; official-source audio only is used",
    }


def audit_audio_payloads(rows: list[dict], payloads: dict[str, bytes]) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    seen_audio = {}
    seen_pcm = {}
    for row in sorted(rows, key=lambda value: value["id"]):
        clip_id = row["id"]
        if row.get("mic") != "mic1" or not row.get("path", "").endswith("_mic1.flac"):
            rejected.append({"id": clip_id, "reason": "non_mic1_audio"})
            continue
        payload = payloads.get(clip_id)
        if payload is None:
            rejected.append({"id": clip_id, "reason": "missing_audio_payload"})
            continue
        if ("size" in row and len(payload) != int(row["size"])) or (
            "crc32" in row and zlib.crc32(payload) != int(row["crc32"])
        ):
            rejected.append({"id": clip_id, "reason": "official_crc_or_size_mismatch"})
            continue
        if len(payload) < 42 or not payload.startswith(b"fLaC"):
            rejected.append({"id": clip_id, "reason": "invalid_audio:missing_flac_streaminfo"})
            continue
        try:
            info = sf.info(io.BytesIO(payload))
        except (RuntimeError, ValueError, OSError) as exc:
            rejected.append({"id": clip_id, "reason": f"invalid_audio_metadata:{type(exc).__name__}:{exc}"})
            continue
        if info.samplerate != 48000 or info.channels != 1:
            rejected.append({"id": clip_id, "reason": "invalid_audio_metadata:sample_rate_or_channel_count"})
            continue
        duration = info.frames / info.samplerate
        if not 1.0 <= duration <= 10.0:
            rejected.append({"id": clip_id, "reason": "duration_out_of_range"})
            continue
        audio_hash = sha256(payload).hexdigest()
        try:
            waveform, duration = preprocess(payload)
        except (RuntimeError, ValueError, OSError) as exc:
            rejected.append({"id": clip_id, "reason": f"invalid_audio:{type(exc).__name__}:{exc}"})
            continue
        pcm_hash = sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()
        duplicate = seen_audio.get(audio_hash) or seen_pcm.get(pcm_hash)
        if duplicate is not None:
            rejected.append({"id": clip_id, "reason": "duplicate_audio_content", "duplicate_of": duplicate})
            continue
        seen_audio[audio_hash] = clip_id
        seen_pcm[pcm_hash] = clip_id
        accepted.append(
            {
                **row,
                "audio_sha256": audio_hash,
                "decoded_pcm16k_sha256": pcm_hash,
                "decoded_pcm_bytes": int(waveform.nbytes),
                "duration_seconds": float(duration),
                "source_sample_rate_hz": 48000,
                "sample_rate_hz": 16000,
                "decoded_samples": int(waveform.size),
            }
        )
    return accepted, rejected


def audit_audio_files(
    rows: list[dict], audio_dir: Path, *, max_local_bytes: int = MAX_LOCAL_DECODED_BYTES
) -> tuple[list[dict], list[dict], dict]:
    """Decode one official FLAC at a time and keep only integrity metadata."""
    accepted = []
    rejected = []
    seen_audio = {}
    seen_pcm = {}
    decoded_bytes = 0
    extracted_bytes = 0
    decode_attempts = 0
    for row in sorted(rows, key=lambda value: value["id"]):
        clip_id = row["id"]
        path = audio_dir / f"{clip_id}.flac"
        if not path.is_file():
            rejected.append({"id": clip_id, "reason": "missing_audio_file"})
            continue
        payload = path.read_bytes()
        extracted_bytes += len(payload)
        decode_attempts += 1
        one_accepted, one_rejected = audit_audio_payloads([row], {clip_id: payload})
        if one_rejected:
            rejected.extend(one_rejected)
            continue
        record = one_accepted[0]
        decoded_bytes += record["decoded_pcm_bytes"]
        if extracted_bytes + decoded_bytes > max_local_bytes:
            raise ValueError("RESOURCE_OVERRIDE_REQUIRED: decoded audio plus local extracted FLAC exceeds P2 cap")
        duplicate = seen_audio.get(record["audio_sha256"]) or seen_pcm.get(record["decoded_pcm16k_sha256"])
        if duplicate is not None:
            rejected.append({"id": clip_id, "reason": "duplicate_audio_content", "duplicate_of": duplicate})
            continue
        seen_audio[record["audio_sha256"]] = clip_id
        seen_pcm[record["decoded_pcm16k_sha256"]] = clip_id
        accepted.append(record)
    return accepted, rejected, {
        "candidate_files_checked": decode_attempts,
        "extracted_flac_bytes": extracted_bytes,
        "decoded_pcm_bytes": decoded_bytes,
        "combined_local_audio_bytes": extracted_bytes + decoded_bytes,
        "audio_sha256_unique": len(seen_audio) == len(accepted),
        "decoded_pcm_sha256_unique": len(seen_pcm) == len(accepted),
    }


def _validate_split_rows(rows: list[dict], split: dict[tuple[str, str], str]) -> dict:
    cell_ids = [(row["speaker"], row["prompt"]) for row in rows]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("duplicate speaker/prompt cell in complete grid")
    if any(row.get("mic") != "mic1" for row in rows):
        raise ValueError("complete grid contains a non-mic1 recording")
    if set(split) != set(cell_ids):
        raise ValueError("split manifest does not partition every grid cell exactly once")
    for key in ("audio_sha256", "decoded_pcm16k_sha256"):
        values = [row.get(key) for row in rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("duplicate audio content across cells")
    support = {}
    for partition, (minimum_speakers, minimum_prompts) in SPLIT_MINIMA.items():
        speaker_counts = Counter(s for (s, _), part in split.items() if part == partition)
        prompt_counts = Counter(p for (_, p), part in split.items() if part == partition)
        if not speaker_counts or not prompt_counts:
            raise ValueError(f"empty {partition} partition")
        if min(speaker_counts.values()) < minimum_speakers or min(prompt_counts.values()) < minimum_prompts:
            raise ValueError(f"{partition} partition fails predeclared speaker/prompt support minimums")
        support[partition] = {
            "min_per_speaker": min(speaker_counts.values()),
            "min_per_prompt": min(prompt_counts.values()),
            "clips": sum(1 for value in split.values() if value == partition),
        }
    return support


def make_split_manifest(rows: list[dict], *, seed: int = SPLIT_SEED) -> dict:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate clip ID or empty source ID in verified audio rows")
    cell_values = [(row["speaker"], row["prompt"]) for row in rows]
    grid = select_max_complete_grid(cell_values)
    selected_cells = {tuple(cell) for cell in grid["cells"]}
    selected_rows = [row for row in rows if (row["speaker"], row["prompt"]) in selected_cells]
    split = balanced_split(grid["speakers"], grid["prompts"], seed, SPLIT_MINIMA)
    support = _validate_split_rows(selected_rows, split)
    row_manifest = [
        {"id": row["id"], "speaker": row["speaker"], "prompt": row["prompt"],
         "partition": split[(row["speaker"], row["prompt"])],
         "audio_sha256": row["audio_sha256"],
         "decoded_pcm16k_sha256": row["decoded_pcm16k_sha256"]}
        for row in sorted(selected_rows, key=lambda value: value["id"])
    ]
    body = {
        "split_seed": seed,
        "speakers": grid["speakers"],
        "prompts": grid["prompts"],
        "cells": len(row_manifest),
        "support_minima": {
            "train_per_speaker": 8,
            "dev_per_speaker": 3,
            "test_per_speaker": 3,
            "train_per_prompt": 12,
            "dev_per_prompt": 5,
            "test_per_prompt": 5,
        },
        "support_observed": support,
        "rows": row_manifest,
    }
    body["sha256"] = sha256(canonical_json(body)).hexdigest()
    return body


def model_rows(rows: list[dict], split_manifest: dict) -> list[dict]:
    expected_hash = split_manifest.get("sha256")
    if not expected_hash:
        raise ValueError("split manifest is missing its SHA256")
    verify_manifest_hash(split_manifest, expected_hash)
    row_by_cell = {(row["speaker"], row["prompt"]): row for row in rows}
    if len(row_by_cell) != len(rows):
        raise ValueError("duplicate cells in model input manifest")
    result = []
    for item in split_manifest["rows"]:
        if item["partition"] not in {"train", "dev", "test"}:
            raise ValueError("unknown partition in split manifest")
        if item["partition"] == "test":
            continue
        key = (item["speaker"], item["prompt"])
        row = row_by_cell.get(key)
        if row is None or row["id"] != item["id"]:
            raise ValueError("split manifest changed or references a missing cell")
        if row.get("audio_sha256") != item["audio_sha256"] or row.get("decoded_pcm16k_sha256") != item["decoded_pcm16k_sha256"]:
            raise ValueError("split audio manifest hash changed")
        result.append({**row, "partition": item["partition"]})
    if any(row["partition"] == "test" for row in result):
        raise ValueError("sealed test row reached model input")
    return result


def verify_historical_micro_test_disjoint(micro_plan: dict, split_manifest: dict) -> dict:
    """Prove the locked P1 train/dev engineering sample does not enter P2's sealed test set."""
    micro_rows = micro_plan.get("rows", [])
    split_rows = split_manifest.get("rows", [])
    micro_ids = [row.get("id") for row in micro_rows]
    if not micro_rows or len(micro_ids) != len(set(micro_ids)):
        raise ValueError("historical P1 micro manifest is empty or has duplicate clip IDs")
    if any(row.get("part") not in {"train", "dev"} for row in micro_rows):
        raise ValueError("historical P1 micro manifest contains a non-train/dev row")
    test_ids = {row["id"] for row in split_rows if row.get("partition") == "test"}
    overlap = sorted(set(micro_ids) & test_ids)
    if overlap:
        raise ValueError(f"historical P1 model access overlaps sealed P2 test IDs: {overlap[:8]}")
    return {"historical_p1_micro_rows": len(micro_rows), "historical_p1_test_overlap_ids": [],
            "overlap_count": 0,
            "P1_micro_plan_sha256": sha256(canonical_json(micro_plan)).hexdigest()}
