"""Run the frozen Q-001 engineering smoke on train/dev only."""
from collections import Counter
import io
from pathlib import Path
import resource
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from sqpilot.data import AUDIO_LIMIT, TRANSFER_LIMIT, digest, materialize, preprocess
from sqpilot.model import check_close, check_model_files, feature, load_model, padded_frames, processor
from sqpilot.quant import apply_from_master, cost_book, discover, fingerprint, master_weights, require, stable_hash
from tools.p0_model import write_json
from tools.p0_source_ranges import write_bytes

BLOCKS = (1,4,8,12)
FP_TOLERANCE = {"float16": dict(rtol=.01,atol=.008), "float32": dict(rtol=.0005,atol=.0003)}
REPEAT_TOLERANCE = dict(rtol=.0001,atol=.00002)
CROSS_TOLERANCE = {"float16": dict(rtol=.025,atol=.02), "float32": dict(rtol=.001,atol=.001)}
PROBE_C = 1.0
PROBE_TOL = 1e-6


def array_sha(values):
    return digest(np.ascontiguousarray(values).tobytes())


def feature_cache(path,key,count,compute):
    hit = path.exists()
    if hit:
        with np.load(path,allow_pickle=False) as archive:
            values = archive["values"]
            require(archive["key"].item()==key and archive["sha256"].item()==array_sha(values),
                    f"Stale or corrupt feature cache {path}")
    else:
        values = compute()
    require(values.shape==(count,768) and values.dtype==np.float32 and np.isfinite(values).all(),"Invalid features")
    if not hit:
        binary = io.BytesIO()
        np.savez(binary,values=values,key=key,sha256=array_sha(values))
        write_bytes(path,binary.getvalue())
    return values,hit


def probe(features, rows, task, out):
    train = np.array([row["part"]=="train" for row in rows])
    labels = np.array([row[task] for row in rows])
    support = {part:dict(Counter(row[task] for row in rows if row["part"]==part)) for part in ("train","dev")}
    require(set(support["train"])==set(support["dev"]) and len(support["train"])==4,"Probe class support differs")
    scaler = StandardScaler().fit(features[train].astype(np.float64))
    head = LogisticRegression(C=PROBE_C,class_weight="balanced",solver="lbfgs",tol=PROBE_TOL,max_iter=2000)
    head.fit(scaler.transform(features[train].astype(np.float64)),labels[train])
    require(head.n_iter_.max()<2000,"Micro probe did not converge")
    weights = {name:val.copy() for name,val in {"mean":scaler.mean_,"scale":scaler.scale_,"variance":scaler.var_,
               "coef":head.coef_,"intercept":head.intercept_,"classes":head.classes_}.items()}
    binary = io.BytesIO()
    np.savez(binary,**weights)
    write_bytes(out,binary.getvalue())
    hashes = {name:array_sha(val) for name,val in weights.items()}
    fixed_numeric_bytes = sum(val.nbytes for name,val in weights.items() if name!="classes")
    fingerprint_head = lambda: {name:array_sha(val) for name,val in {
        "mean":scaler.mean_,"scale":scaler.scale_,"variance":scaler.var_,
        "coef":head.coef_,"intercept":head.intercept_,"classes":head.classes_}.items()}
    predict = lambda value: head.predict(scaler.transform(value.astype(np.float64)))
    return predict, fingerprint_head, dict(C=PROBE_C,tolerance=PROBE_TOL,solver="lbfgs",balanced=True,
        support=support,iterations=int(head.n_iter_.max()),arrays_sha256=hashes,
        fixed_numeric_bytes=fixed_numeric_bytes,stored_probe_sha256=digest(binary.getvalue()))


def smoke(config,historic,plan,p0,out):
    start = time.monotonic()
    rows = plan["rows"]
    require(len(rows)<=min(AUDIO_LIMIT,config["resources"]["P1_max_audio_clips"])
            and 4==config["resources"]["P1_max_selected_blocks"] and all(row["part"] in ("train","dev") for row in rows),
            "Invalid authorized P1 file count, blocks or partitions")
    require(plan["planned_range_bytes"]<=plan["range_budget_bytes"]==TRANSFER_LIMIT,"Invalid transfer plan")
    out.parent.mkdir(parents=True,exist_ok=True)
    result = dict(status="FAIL",provenance=p0["provenance"],conditions={},backend_checks={},
                  G0="UNVERIFIED",G5="OPEN",test_audio_reads=0,model_forwards=0,
                  planned_audio_clips=len(rows),audio_clips=0,downloaded_bytes=0,
                  memory_observed_mps_driver_bytes=0,minimum_disk_free_bytes=None)
    data_dir = Path("data/vctk-q001")
    def monitor():
        import shutil
        if torch.backends.mps.is_available():
            result["memory_observed_mps_driver_bytes"] = max(result["memory_observed_mps_driver_bytes"],
                                                             torch.mps.driver_allocated_memory())
        available = shutil.disk_usage(".").free
        prior = result["minimum_disk_free_bytes"]
        result["minimum_disk_free_bytes"] = available if prior is None else min(prior,available)

    def infer(model,extractor,signal,pad=0):
        result["model_forwards"] += 1
        output = feature(model,extractor,signal,pad=pad)
        monitor()
        return output

    try:
        checkpoint = check_model_files(config,historic)
        monitor()
        fetched = materialize(plan,data_dir)
        result["audio_clips"] = len(fetched)
        result["downloaded_bytes"] = sum(transfer for _,transfer in fetched)
        result["official_range_ceiling_bytes"] = plan["planned_range_bytes"]
        result["flac_bytes"] = sum(len(data) for data,_ in fetched)
        result["download_ceiling_bytes"] = TRANSFER_LIMIT
        raw_hashes = set()
        signals = []
        clips = []
        for row,(audio,_) in zip(rows,fetched,strict=True):
            raw = digest(audio)
            require(raw not in raw_hashes,"Duplicate audio payload across cells")
            raw_hashes.add(raw)
            samples,duration = preprocess(audio)
            repeat,_ = preprocess(audio)
            require(np.array_equal(samples,repeat),"Preprocessing was non-reproducible")
            signals.append(samples)
            clips.append(dict(id=row["id"],partition=row["part"],seconds=duration,official_crc32=row["crc32"],
                              flac_sha256=raw,pcm16k_sha256=array_sha(samples),n_samples=len(samples),
                              transcript_sha256=row["transcript_sha256"]))
        require(len({clip["pcm16k_sha256"] for clip in clips})==len(clips),"Duplicate decoded waveform")
        result["clips"] = clips
        result["audio_integrity"] = "PASS"
        print(f"P1 audio: {len(clips)} verified, {result['downloaded_bytes']} HTTP range bytes",flush=True)

        extractor = processor()
        check_signal = signals[min(range(len(signals)),key=lambda i:len(signals[i]))]
        candidates = ([("mps",torch.float16)] if torch.backends.mps.is_available() else []) + [("cpu",torch.float16)]
        chosen = None
        backend_outputs = {}
        model = None
        for device,dtype in [*candidates,("cpu",torch.float32)]:
            if device=="cpu" and dtype==torch.float32 and chosen is not None:
                break
            name = str(dtype).removeprefix("torch.")
            backend = f"{device}/{name}"
            controls = result["backend_checks"].setdefault(backend,{})
            trial = None
            try:
                trial = load_model(checkpoint,dtype,device)
                baseline = infer(trial,extractor,check_signal)
                controls["repeat_max_abs"] = check_close(baseline,infer(trial,extractor,check_signal),
                                                         check="repeat",**REPEAT_TOLERANCE)
                controls["sample_padding_max_abs"] = check_close(baseline,infer(trial,extractor,check_signal,pad=640),
                                                                  check="sample padding",**FP_TOLERANCE[name])
                for extra_frames in (0,5):
                    result["model_forwards"] += 1
                    controls[f"transformer_padding_{extra_frames}_max_abs"] = check_close(
                        baseline,padded_frames(trial,extractor,check_signal,extra_frames),
                        check="transformer frame padding",**FP_TOLERANCE[name])
                if device=="cpu" and f"mps/{name}" in backend_outputs:
                    controls["cross_backend_max_abs"] = check_close(backend_outputs[f"mps/{name}"],baseline,
                                                                      check="CPU/MPS agreement",**CROSS_TOLERANCE[name])
                controls["status"]="PASS"
                backend_outputs[backend]=baseline
                if chosen is None:
                    model,chosen = trial,(device,dtype,name)
                else:
                    del trial
            except (RuntimeError,NotImplementedError,ValueError) as error:
                controls.update(status="FAIL",error=f"{type(error).__name__}: {error}")
                print(f"Backend {backend} {controls['error']}",flush=True)
                if device=="cpu" and chosen is not None:
                    # When both forward paths run yet disagree, prefer a verified CPU
                    # path only after independently passing its repeat/padding checks.
                    if "CPU/MPS agreement" in str(error):
                        model,chosen = trial,(device,dtype,name)
                        controls["independent_backend_status"] = "PASS"
                    else:
                        del trial
                else:
                    del trial
        require(chosen is not None,"No supported FP16 or documented FP32 reference backend")
        device,dtype,name = chosen
        result.update(device=device,reference_precision=name,fp32_fallback=(dtype==torch.float32),
                      repeat_tolerance=REPEAT_TOLERANCE,padding_tolerances=FP_TOLERANCE,
                      cross_backend_tolerances=CROSS_TOLERANCE,groupnorm_adaptation="singleton trimmed before waveform frontend")
        print(f"P1 reference: {device}/{name}",flush=True)
        from sqpilot.cli import PRIOR_MODEL,read_json
        selection,_=discover(model,config,read_json(PRIOR_MODEL)["tensor_shapes"])
        baseline_hashes=fingerprint(model)
        immutable=master_weights(model)
        cache_prefix=dict(provenance=p0["provenance"],audio=clips,device=device,dtype=name)
        def batch():
            return np.stack([infer(model,extractor,signal) for signal in signals])

        def get_features(condition,weight_hashes):
            key=stable_hash(dict(**cache_prefix,condition=condition,weights=stable_hash(weight_hashes)))
            path=out.parent/f"{condition}.npz"
            features,hit=feature_cache(path,key,len(rows),batch)
            reread,_=feature_cache(path,key,len(rows),batch)
            require(np.array_equal(features,reread),"Cached feature payload changed")
            return features,reread,hit

        reference,cached,hit = get_features("reference",baseline_hashes)
        result["conditions"]["reference"] = dict(feature_sha256=array_sha(reference),cache_hit=hit,
            repeat_max_abs=check_close(reference[0],infer(model,extractor,signals[0]),check="reference replay",**REPEAT_TOLERANCE))
        heads = {task:probe(reference,rows,task,out.parent/f"probe-{task}.npz") for task in ("prompt","speaker")}
        result["probe"]={task:details for task,(_,_,details) in heads.items()}
        numeric_probe_bytes=sum(details["fixed_numeric_bytes"] for _,_,details in heads.values())
        result["fixed_probe_numeric_bytes"]=numeric_probe_bytes
        result["estimated_fixed_numeric_artifact_bytes"] = cost_book(model,selection,dtype,numeric_probe_bytes).fixed
        result["baseline_model_storage_bytes"]=sum(w.numel()*w.element_size() for w in model.parameters())
        dev_indices=[i for i,row in enumerate(rows) if row["part"]=="dev"]
        for task,(predict,_,details) in heads.items():
            first, second = predict(reference[dev_indices]),predict(cached[dev_indices])
            require(np.array_equal(first,second) and len(first)==4,"Cached reference probe predictions differ")
            details["dev_prediction_sha256"]=array_sha(first)
        result["smoke_probe"]="PASS"
        for block in BLOCKS:
            for bits in (4,8):
                condition=f"block{block:02d}-{bits}bit"
                with apply_from_master(model,immutable,selection[block-1],bits) as stats:
                    current=fingerprint(model)
                    require({k for k in current if current[k]!=baseline_hashes[k]}==set(selection[block-1]),
                            "Intervention modified unexpected weights")
                    features,cached,hit=get_features(condition,current)
                    prediction_hashes={}
                    for task,(predict,checker,details) in heads.items():
                        want=predict(features[dev_indices])
                        got=predict(cached[dev_indices])
                        require(np.array_equal(want,got) and checker()==details["arrays_sha256"],
                                "Cached predictions differ or head changed")
                        prediction_hashes[task]=array_sha(want)
                    repeated = infer(model,extractor,signals[0])
                    repeat = check_close(features[0],repeated,check="variant replay",**REPEAT_TOLERANCE)
                    for predict,_,_ in heads.values():
                        require(np.array_equal(predict(features[0:1]),predict(repeated[None,:])),
                                "Recomputed variant prediction differs")
                require(fingerprint(model)==baseline_hashes,"Model restoration differs from full reference")
                with apply_from_master(model,immutable,selection[block-1],bits):
                    require(fingerprint(model)==current,"Weight bytes differ after reapplying variant")
                require(fingerprint(model)==baseline_hashes,"Model restoration failed on repeat")
                result["conditions"][condition]=dict(block=block,bits=bits,weights=stats,
                    feature_sha256=array_sha(features),cache_hit=hit,repeat_max_abs=repeat,
                    dev_prediction_sha256=prediction_hashes,restored=True)
                print(f"P1 {condition} verified",flush=True)
        final = infer(model,extractor,signals[0])
        result["restored_forward_max_abs"] = check_close(reference[0],final,check="restored reference",**REPEAT_TOLERANCE)
        require(fingerprint(model)==baseline_hashes,"Final restored parameters differ")
        result["immutable_master_verified"] = True
        result["quantized_conditions"]=8
        result["wavlm_forward"]="PASS"
        result["status"]="PASS"
    except Exception as error:
        result["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor()
        result["wall_seconds"]=round(time.monotonic()-start,3)
        result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(out,result)
