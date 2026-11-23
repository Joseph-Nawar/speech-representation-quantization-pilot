"""Pinned WavLM loading and mask-aware inference."""
from collections import OrderedDict
from contextlib import ExitStack
from hashlib import file_digest
import inspect
from pathlib import Path

import numpy as np
import torch
from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor

from sqpilot.quant import require

REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LFS_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
SOURCE_HASH = "0d3fb6abe51dc4c4f019ee0e5907f653eb602adc525f5638024068389ed18098"
MODEL_DIR = Path("cache/p0-model")
ALIASES = {"encoder.pos_conv_embed.conv.weight_g":"encoder.pos_conv_embed.conv.parametrizations.weight.original0",
           "encoder.pos_conv_embed.conv.weight_v":"encoder.pos_conv_embed.conv.parametrizations.weight.original1"}


def file_sha(path):
    with open(path,"rb") as file:
        return file_digest(file,"sha256").hexdigest()


def check_model_files(config, historic):
    require(config["model"]["model_id"] == "microsoft/wavlm-base-plus"
            and config["model"]["revision"] == config["model"]["processor_revision"] == REVISION, "Model revision changed")
    for name in ("config.json","preprocessor_config.json","README.md"):
        path = MODEL_DIR/name
        require(file_sha(path)==historic[str(path)],f"Changed model document {name}")
    require(file_sha(inspect.getfile(WavLMModel))==SOURCE_HASH,"Unverified WavLM implementation")
    checkpoint = MODEL_DIR/"pytorch_model.bin"
    require(checkpoint.stat().st_size==377617425 and file_sha(checkpoint)==LFS_SHA256,"Pinned checkpoint mismatch")
    return checkpoint


def load_model(checkpoint, dtype, device):
    config = WavLMConfig.from_pretrained(MODEL_DIR,local_files_only=True)
    with torch.device("meta"):
        model = WavLMModel(config)
    original = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    mapped = OrderedDict((ALIASES.get(name,name), value) for name,value in original.items())
    require(len(mapped)==len(original),"Checkpoint key alias collision")
    model.load_state_dict(mapped,strict=True,assign=True)
    model.eval().requires_grad_(False)
    model.config.apply_spec_augment = False
    require(not model.training and not any(p.requires_grad for p in model.parameters()),"Encoder is not frozen")
    return model.to(device=device,dtype=dtype)


def processor():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR,local_files_only=True)
    require(extractor.sampling_rate==16000 and extractor.return_attention_mask and not extractor.do_normalize
            and extractor.padding_side=="right" and extractor.padding_value==0,"Processor configuration differs")
    return extractor


def masked_mean(last_four, mask):
    require(len(last_four)==4 and mask.dtype==torch.bool and all(state.shape[:2]==mask.shape for state in last_four)
            and mask.any(dim=1).all().item(),"Invalid hidden states or frame mask")
    values = torch.stack(last_four).float().mean(dim=0).masked_fill(~mask.unsqueeze(-1),0)
    pooled = values.sum(dim=1) / mask.sum(dim=1,keepdim=True)
    require(torch.isfinite(pooled).all().item(),"Non-finite WavLM features")
    return pooled.cpu().numpy()


@torch.inference_mode()
def feature(model, extractor, waveform, pad=0):
    parameter = next(model.parameters())
    inputs = extractor(waveform,sampling_rate=16000,return_tensors="pt",return_attention_mask=True,
                       padding="max_length" if pad else False,max_length=len(waveform)+pad)
    mask = inputs.attention_mask
    n = int(mask.sum())
    require(n==len(waveform) and torch.all(mask[0,:n]==1) and torch.all(mask[0,n:]==0),"Invalid right-padded sample mask")
    # GroupNorm aggregates over time. Remove padding before the waveform frontend.
    # Transformer padding invariance is checked separately with actual projected frames.
    sample = inputs.input_values[:,:n].to(device=parameter.device,dtype=parameter.dtype)
    mask = mask[:,:n].to(device=parameter.device)
    output = model(sample,attention_mask=mask,output_hidden_states=True,return_dict=True)
    last_four = output.hidden_states[-4:]
    frame_mask = model._get_feature_vector_attention_mask(last_four[-1].shape[1],mask,add_adapter=False)
    return masked_mean(last_four,frame_mask)[0]


@torch.inference_mode()
def padded_frames(model, extractor, waveform, extra_frames):
    """Test padding after the time-normalized frontend through all attention blocks."""
    parameter = next(model.parameters())
    waveform = extractor(waveform,sampling_rate=16000,return_tensors="pt").input_values
    waveform = waveform.to(device=parameter.device,dtype=parameter.dtype)
    projected,_ = model.feature_projection(model.feature_extractor(waveform).transpose(1,2))
    length = projected.shape[1]
    padded = torch.nn.functional.pad(projected,(0,0,0,extra_frames))
    mask = torch.arange(length+extra_frames,device=parameter.device).unsqueeze(0)<length
    layer_outputs = []
    with ExitStack() as callbacks:
        for layer in model.encoder.layers[-4:]:
            hook = layer.register_forward_hook(lambda _layer,_args,output: layer_outputs.append(output[0]))
            callbacks.callback(hook.remove)
        model.encoder(padded,attention_mask=mask)
    return masked_mean(layer_outputs,mask)[0]


def check_close(a,b,*,rtol,atol,check):
    diff = float(np.max(np.abs(a-b)))
    require(np.allclose(a,b,rtol=rtol,atol=atol), f"{check} max absolute difference {diff:.6g}")
    return diff
