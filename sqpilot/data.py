"""Metadata-only split and bounded official VCTK materialization."""
from collections import Counter
import concurrent.futures
from fractions import Fraction
from hashlib import sha256
import io
from itertools import combinations, permutations, product
import json
from pathlib import Path
import re
import struct
import zlib

from scipy.signal import resample_poly
import numpy as np
import soundfile as sf

from tools.p0_source_ranges import OFFICIAL, ROOT, Remote, norm, write_bytes
from sqpilot.quant import require, stable_hash

PARTS = ("train", "dev", "test")
AUDIO_LIMIT = 128
TRANSFER_LIMIT = 16 * 1024**2
MEMBER_LIMIT = 1024**2
OFFICIAL_SIZE = 11747302977
OFFICIAL_ETAG = '"8a6ba2946b36fcbef0212cad601f4bfa"'
FLAC_STREAMINFO_BYTES = 42


def digest(data):
    return sha256(data).hexdigest()


def rank(seed, *ids):
    return sha256(json.dumps((*ids, seed), separators=(",", ":")).encode()).digest()


def balanced_split(speakers, prompts, seed, minima):
    """Allocate exact rounded totals by hash, then deterministically repair support."""
    require(len(set(speakers)) == len(speakers) and len(set(prompts)) == len(prompts), "Duplicate grid IDs")
    require(speakers and prompts, "Empty grid")
    cells = sorted(product(speakers, prompts), key=lambda xy: (rank(seed, *xy), xy))
    ratios = dict(zip(PARTS, map(Fraction, ("0.60", "0.20", "0.20")), strict=True))
    fractional = {part: len(cells)*ratios[part] for part in PARTS}
    totals = {part: int(fractional[part]) for part in PARTS}
    for part in sorted(PARTS, key=lambda p: (-(fractional[p]-totals[p]), PARTS.index(p)))[:len(cells)-sum(totals.values())]:
        totals[part] += 1
    labels = [part for part in PARTS for _ in range(totals[part])]
    split = dict(zip(cells, labels, strict=True))

    # First repair speaker support using cross-speaker swaps. Then swap inside
    # individual speakers to repair prompt support without undoing speaker balance.
    for axis, ids in enumerate((sorted(speakers), sorted(prompts))):
        by_id = {label: [cell for cell in cells if cell[axis] == label] for label in ids}
        counts = {label: Counter(split[cell] for cell in own) for label,own in by_id.items()}
        for label in ids:
            for target in PARTS:
                deficit = minima[target][axis] - counts[label][target]
                for _ in range(max(0, deficit)):
                    swap = next(
                        ((cell, donor) for cell in by_id[label] if counts[label][split[cell]] > minima[split[cell]][axis]
                         for donor in cells if donor[axis] != label and split[donor] == target
                         and counts[donor[axis]][target] > minima[target][axis]
                         and (axis == 0 or cell[0] == donor[0])),
                        None,
                    )
                    require(swap is not None, f"No deterministic minimum-support repair at axis {axis}")
                    cell, donor = swap
                    original = split[cell]
                    split[cell], split[donor] = target, original
                    counts[label].update({target: 1, original: -1})
                    counts[donor[axis]].update({target: -1, original: 1})
        require(all(counts[label][part] >= minima[part][axis] for label in ids for part in PARTS),
                f"Invalid repaired split at axis {axis}")
    require(Counter(split.values()) == totals, "Split totals changed during repair")
    return split


def validate_micro(micro, split):
    cells = [(s,p) for s,p,_ in micro]
    require(0 < len(cells) <= AUDIO_LIMIT and len(cells) == len(set(cells)), "Duplicate micro or invalid clip count")
    require(all(part in ("train","dev") and split.get((s,p)) == part for s,p,part in micro),
            "Micro selection would access sealed or ineligible recording")
    for axis in (0,1):
        for part,count in (("train",2),("dev",1)):
            support = Counter(row[axis] for row in micro if row[2] == part)
            require(len(support) == 4 and set(support.values()) == {count}, "Micro class support changed")


def select_micro(split, seed):
    """Select four speakers and prompts with two train and one dev per class."""
    speakers = sorted({s for s,_ in split}, key=lambda value:(rank(seed,"speaker",value),value))
    prompts = sorted({p for _,p in split}, key=lambda value:(rank(seed,"prompt",value),value))
    for speakers4 in combinations(speakers, 4):
        usable = [p for p in prompts if any(split[s,p]=="dev" for s in speakers4)
                  and sum(split[s,p]=="train" for s in speakers4) >= 2]
        for prompts4 in combinations(usable, 4):
            train_options = [tuple(combinations([p for p in prompts4 if split[s,p]=="train"],2)) for s in speakers4]
            if not all(train_options):
                continue
            dev = next((perm for perm in permutations(prompts4) if all(split[s,p]=="dev" for s,p in zip(speakers4,perm))), None)
            if dev is None:
                continue
            train = next((sets for sets in product(*train_options)
                          if Counter(p for own in sets for p in own) == dict.fromkeys(prompts4,2)), None)
            if train is not None:
                micro = sorted([(s,p,"train") for s, own in zip(speakers4,train) for p in own]
                               + [(s,p,"dev") for s,p in zip(speakers4,dev)])
                validate_micro(micro, split)
                return micro
    raise ValueError("No balanced train/dev micro subset in candidate split")


def plan_micro(config, historic_hashes):
    files = ("candidate-grid-audit.json","official-index.json","official-texts.json")
    frozen = {}
    for file in files:
        path = ROOT / file
        contents = path.read_bytes()
        require(digest(contents) == historic_hashes[str(path)], f"Changed P0 evidence {path}")
        frozen[file] = json.loads(contents)
    candidate = frozen[files[0]]
    speakers,prompts = candidate["speakers"],candidate["prompts"]
    require((len(speakers),len(prompts)) == (63,18) and all(re.fullmatch(r"(?:p\d+|s\d+)",s) for s in speakers)
            and all(re.fullmatch(r"\d{3}",p) for p in prompts), "Unexpected candidate grid")
    source = config["data"]
    require(source["microphone"]=="mic1" and source["split"]==dict(zip(PARTS,(.6,.2,.2)))
            and source["sample_rate_hz"]==16000 and source["duration_seconds"]==[1.0,10.0], "Wrong data settings")
    minima = {part:(source[f"{part}_min_per_speaker"],source[f"{part}_min_per_prompt"]) for part in PARTS}
    seed = source["split_seed"]
    split = balanced_split(speakers,prompts,seed,minima)
    micro = select_micro(split,seed)
    require(len(micro) <= config["resources"]["P1_max_audio_clips"], "Configured ceiling exceeded")
    index = sorted(frozen[files[1]], key=lambda member:member[-1])
    needed = {f"wav48_silence_trimmed/{s}/{s}_{p}_mic1.flac":(s,p,part) for s,p,part in micro}
    texts = frozen[files[2]]
    canonical = {}
    rows = []
    for (path,crc,size,compressed,offset), next_member in zip(index,index[1:]):
        if path not in needed:
            continue
        s,p,part = needed.pop(path)
        clip_id = f"{s}_{p}_mic1"
        text = norm(texts[f"{s}_{p}"])
        require(text and canonical.setdefault(p,text)==text, "Inconsistent canonical prompt transcript")
        span = next_member[-1] - offset
        require(0 < compressed <= span <= MEMBER_LIMIT and 0 < size <= MEMBER_LIMIT, "Member exceeds bounds")
        rows.append(dict(id=clip_id,speaker=s,prompt=p,part=part,path=path,crc32=crc,
                         size=size,compressed=compressed,offset=offset,span=span,transcript_sha256=digest(text.encode())))
    require(not needed and len(rows) == len(micro), "Missing official members")
    planned = sum(row["span"] for row in rows)
    require(planned <= TRANSFER_LIMIT, "Planned transfer exceeds ceiling")
    return dict(study_id=config["study_id"], split_seed=seed, candidate_sha256=historic_hashes[str(ROOT/files[0])],
                preview_split_sha256=stable_hash(sorted((s,p,v) for (s,p),v in split.items())),
                split_totals=dict(Counter(split.values())), range_budget_bytes=TRANSFER_LIMIT,
                planned_range_bytes=planned, rows=sorted(rows,key=lambda row:row["id"]))


def unpack_member(data, row):
    require(len(data) == row["span"], "Short ZIP member")
    signature,_,flags,method,*_,name_length,extra = struct.unpack_from("<4s5H3L2H",data)
    require(signature == b"PK\x03\x04" and not flags & 1 and method in (0,8), "Unexpected ZIP header")
    path = data[30:30+name_length].decode()
    require(path == row["path"], "ZIP local header path mismatch")
    start = 30+name_length+extra
    require(start+row["compressed"]<=len(data), "ZIP range omitted compressed member data")
    payload = data[start:start+row["compressed"]]
    if method == 8:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(payload,row["size"]+1)
        require(decoder.eof and not decoder.unconsumed_tail, "Invalid deflated ZIP member")
    require(len(payload)==row["size"] and zlib.crc32(payload)==row["crc32"], "Official member CRC/size mismatch")
    return payload


def materialize(plan, directory):
    directory.mkdir(parents=True,exist_ok=True)
    require(len(plan["rows"]) <= AUDIO_LIMIT, "Audio clip count exceeds ceiling")
    def fetch(row):
        path = directory/f'{row["id"]}.flac'
        if path.exists():
            data = path.read_bytes()
            require(len(data)==row["size"] and zlib.crc32(data)==row["crc32"], "Stale local audio")
            transferred = 0
        else:
            source = Remote(OFFICIAL,OFFICIAL_SIZE,OFFICIAL_ETAG)
            data = unpack_member(source.pull(row["offset"],row["span"]), row)
            write_bytes(path,data)
            transferred = source.bytes
        return data, transferred
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch,plan["rows"]))


def preprocess(data):
    require(len(data)>=FLAC_STREAMINFO_BYTES and data.startswith(b"fLaC"),"Expected FLAC member")
    wave,rate = sf.read(io.BytesIO(data),dtype="float32",always_2d=True)
    seconds = len(wave)/rate
    require(rate==48000 and wave.shape[1]==1 and 1<=seconds<=10 and np.isfinite(wave).all(),
            "Invalid source sample rate/channel/length/values")
    mono = wave[:,0]
    require(np.any(mono != 0) and np.ptp(mono)>0,"Silent or constant waveform")
    samples = np.ascontiguousarray(resample_poly(mono, 1, 3),dtype=np.float32)
    require(np.isfinite(samples).all() and np.ptp(samples)>0 and len(samples)==(len(mono)+2)//3,
            "Invalid resampling")
    return samples, seconds
