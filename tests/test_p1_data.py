from collections import Counter
import pytest

from sqpilot.data import balanced_split, select_micro, validate_micro

SPEAKERS = [f"p{i:03}" for i in range(63)]
PROMPTS = [f"{i:03}" for i in range(18)]
MINIMA = {"train": (8, 12), "dev": (3, 5), "test": (3, 5)}


def test_balanced_split_is_deterministic_complete_and_micro_never_uses_test():
    split = balanced_split(SPEAKERS, PROMPTS, 20261122, MINIMA)
    assert split == balanced_split(SPEAKERS[::-1], PROMPTS[::-1], 20261122, MINIMA)
    assert Counter(split.values()) == {"train": 680, "dev": 227, "test": 227}
    for part, (per_speaker, per_prompt) in MINIMA.items():
        assert min(Counter(s for (s,p), v in split.items() if v == part).values()) >= per_speaker
        assert min(Counter(p for (s,p), v in split.items() if v == part).values()) >= per_prompt
    micro = select_micro(split, 20261122)
    validate_micro(micro, split)
    assert len(micro) == 12


def test_reject_duplicate_ids_and_sealed_samples():
    with pytest.raises(ValueError, match="Duplicate grid IDs"):
        balanced_split(SPEAKERS + SPEAKERS[:1], PROMPTS, 20261122, MINIMA)
    split = balanced_split(SPEAKERS, PROMPTS, 20261122, MINIMA)
    micro = select_micro(split, 20261122)
    test_cell = next(pair for pair, part in split.items() if part == "test")
    with pytest.raises(ValueError, match="seal"):
        validate_micro(micro + [(*test_cell, "test")], split)
    with pytest.raises(ValueError, match="Duplicate micro"):
        validate_micro(micro + micro[:1], split)


def test_range_member_checks_identity_and_crc_even_on_correct_length():
    import io
    import zipfile
    from sqpilot.data import unpack_member

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        path = "wav48_silence_trimmed/p999/p999_001_mic1.flac"
        bundle.writestr(path,b"synthetic member")
        info = bundle.getinfo(path)
        span = archive.tell()
    row = dict(path=path,crc32=info.CRC,size=info.file_size,compressed=info.compress_size,span=span)
    assert unpack_member(archive.getvalue()[:span],row)==b"synthetic member"
    with pytest.raises(ValueError,match="CRC"):
        unpack_member(archive.getvalue()[:span],dict(row,crc32=0))
    with pytest.raises(ValueError,match="path"):
        unpack_member(archive.getvalue()[:span],dict(row,path="another-file"))


def test_flac_preprocessing_has_repeatable_16khz_shape_and_rejects_silence():
    import io
    import soundfile as sf
    import numpy as np
    from sqpilot.data import preprocess

    def flac(values):
        buffer = io.BytesIO()
        sf.write(buffer,values,48000,format="FLAC",subtype="PCM_16")
        return buffer.getvalue()

    source=flac((np.arange(48000)%64).astype(np.float32)/100)
    first,seconds=preprocess(source)
    assert len(first)==16000 and seconds==1
    assert np.array_equal(first,preprocess(source)[0])
    with pytest.raises(ValueError,match="Silent"):
        preprocess(flac(np.zeros(48000)))
