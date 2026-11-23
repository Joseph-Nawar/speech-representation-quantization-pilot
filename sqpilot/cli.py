"""Q-001 preflight and micro-smoke entry point."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import torch
import yaml

from sqpilot.data import plan_micro
from sqpilot.model import LFS_SHA256, check_model_files, file_sha, load_model
from sqpilot.quant import apply_from_master, cost_book, discover, fingerprint, master_weights, require, stable_hash
from tools.p0_model import write_json

CONFIG = Path("configs/pilot.yaml")
AMENDMENT = Path("docs/amendments/Q001_SCOPE.md")
HISTORIC = Path("reports/P0_STATUS_V1.json")
PRIOR_MODEL = Path("runs/P0/model.json")
STUDY = "speech-quant-diagnostic-20261122-v2"


def read_json(path):
    return json.loads(path.read_bytes())


def inputs():
    config = yaml.safe_load(CONFIG.read_bytes())
    require((config["study_id"],config["protocol_version"],config["amendment"])==(STUDY,2,"Q-001"),"Unexpected study")
    previous = read_json(HISTORIC)
    require(previous["P0"]["audio_clips"]==previous["P0"]["model_forwards"]==previous["P0"]["variants"]
            ==previous["P1"]["test_reads"]==0 and not previous["P1"]["attempted"],"Historical P0 integrity")
    historic = previous["sha256"]["ignored_local_evidence"]
    require(file_sha(PRIOR_MODEL)==historic[str(PRIOR_MODEL)],"Changed original model inventory")
    return config,historic


def provenance(config, plan):
    paths = ["sqpilot", "tests", str(CONFIG), str(AMENDMENT), "requirements-p0-observed.txt"]
    pending = subprocess.check_output(["git","status","--porcelain","--",*paths],text=True)
    require(not pending, "Commit the stage implementation and amended configuration before running the stage")
    commit = subprocess.check_output(["git","log","-1","--format=%H","--",*paths],text=True).strip()
    require(len(commit)==40,"Cannot identify runtime code revision")
    return dict(study_id=STUDY,protocol_version=2,amendment="Q-001",
                scope_sha256=stable_hash(config["quantization"]),config_sha256=file_sha(CONFIG),
                amendment_sha256=file_sha(AMENDMENT),model_revision=config["model"]["revision"],
                processor_revision=config["model"]["processor_revision"],checkpoint_sha256=LFS_SHA256,
                candidate_sha256=plan["candidate_sha256"],micro_plan_sha256=stable_hash(plan),
                software_sha256=file_sha("requirements-p0-observed.txt"),
                code_commit=commit)


def freeze(path, content):
    if path.exists():
        require(read_json(path)==content, f"Frozen artifact changed: {path}")
    else:
        write_json(path,content)


def p0(config,historic,plan,args):
    start = time.monotonic()
    checkpoint = check_model_files(config,historic)
    model = load_model(checkpoint,torch.float32,"cpu")
    selection,rows = discover(model,config,read_json(PRIOR_MODEL)["tensor_shapes"])
    require(sum(row["selected"] for row in rows)==72 and len(rows)==84,"Scope reconciliation failed")
    all_before = fingerprint(model)
    master = master_weights(model)
    seven = tuple(tuple((*names,f"encoder.layers.{i}.attention.gru_rel_pos_linear.weight"))
                  for i,names in enumerate(selection))
    budgets = {}
    for dtype in (torch.float16,torch.float32):
        expected = read_json(PRIOR_MODEL)["interpretations"]["six_named"]["by_baseline_dtype"]
        old = cost_book(model,seven,dtype)
        new = cost_book(model,selection,dtype)
        label = str(dtype).removeprefix("torch.")
        fp = {"float16":"fp16","float32":"fp32"}[label]
        fractions = config["quantization"]["budgets_fraction_between_uniform4_and_uniform8"]
        lo,hi = new.total((4,)*len(selection)),new.total((8,)*len(selection))
        require(lo==expected[fp]["uniform4_calculated_total_bytes"] and hi==expected[fp]["uniform8_calculated_total_bytes"],
                "Calculated endpoint changed beyond Q-001")
        require(lo<hi and fractions==[.25,.50,.75] and config["quantization"]["budget_cost_tolerance_percent"]==.5,
                "Budget settings changed")
        entries = {}
        previous = lo
        for fraction in fractions:
            ceiling = new.budget(fraction)
            feasible = new.feasible(fraction)
            achieved = max(map(new.total,feasible))
            require(previous<=ceiling<=hi and achieved<=ceiling,"Invalid feasible budget")
            entries[str(fraction)] = dict(ceiling_bytes=int(ceiling),achieved_max_bytes=achieved,
                                          feasible_mappings=len(feasible),prior_seven_linear_ceiling=int(old.budget(fraction)))
            previous = ceiling
        budgets[fp] = dict(fixed_bytes=new.fixed, per_block_bytes=new.blocks, all4_bytes=lo,all8_bytes=hi,
                           prior_seven_linear_all4=old.total((4,)*len(selection)),
                           prior_seven_linear_all8=old.total((8,)*len(selection)),fractions=entries)
    perturbations = {}
    for bits in (4,8):
        first = None
        for _ in range(2):
            with apply_from_master(model,master,selection[0],bits) as stats:
                during = fingerprint(model)
                change = {key for key in during if during[key]!=all_before[key]}
                require(change==set(selection[0]),"P0 actual intervention modified unexpected parameters")
                require(first is None or during==first,"Quantization was not byte-repeatable")
                first = during
                perturbations[str(bits)] = stats
            require(fingerprint(model)==all_before,"P0 actual model restoration failed")
    parameters = [{"path":name,"shape":list(value.shape),"dtype":str(value.dtype),"parameters":value.numel()}
                  for name,value in model.named_parameters()]
    require(len(parameters)==len(all_before),"Checkpoint parameter count changed")
    output = dict(status="PASS",provenance=provenance(config,plan),inventory=rows,all_parameters=parameters,
                  model_budget_estimates=budgets,model_sha256=stable_hash(all_before),p0_weight_perturbation=perturbations,
                  audio_clips=0,model_forwards=0,device="cpu",wall_seconds=round(time.monotonic()-start,3),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,G0="UNVERIFIED",G5="OPEN")
    write_json(args.out,output)
    print(f"P0 PASS. 84 audited matrices, 72 eligible and 12 excluded gate weights. {output['wall_seconds']} seconds.",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","p0","p1"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--plan",type=Path,default=Path("runs/P0_Q001/micro-plan.json"))
    parser.add_argument("--p0", type=Path,default=Path("runs/P0_Q001/p0.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261122)
    config,historic = inputs()
    plan = plan_micro(config,historic)
    if args.stage=="plan":
        freeze(args.out,plan)
        print(f"Frozen micro plan: {len(plan['rows'])} train/dev-only clips, {plan['planned_range_bytes']} bytes ceiling for requested ranges",flush=True)
        return
    require(read_json(args.plan)==plan,"Frozen micro plan changed")
    if args.stage=="p0":
        p0(config,historic,plan,args)
    else:
        from sqpilot.smoke import smoke
        stage0 = read_json(args.p0)
        require(stage0["status"]=="PASS" and stage0["provenance"]==provenance(config,plan),
                "P1 requires a matching P0 pass and frozen code")
        smoke(config,historic,plan,stage0,args.out)


if __name__=="__main__":
    main()
