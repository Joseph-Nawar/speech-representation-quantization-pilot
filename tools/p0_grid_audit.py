"""Cross-check all mirror metadata against the official ZIP and find a complete grid."""
import collections
import hashlib
import itertools
import json
import math
import re

from p0_source_ranges import ROOT, norm, write_bytes

index=json.loads((ROOT/'official-index.json').read_bytes())
texts=json.loads((ROOT/'official-texts.json').read_bytes())
audio={}
for path,crc,size,*_ in index:
    match=re.fullmatch(r'wav48_silence_trimmed/([^/]+)/(\1_\d+)_(mic[12])\.flac',path)
    if match:
        speaker,uid,mic=match.groups()
        k=uid,mic
        if k in audio: raise ValueError('Duplicate official recording')
        audio[k]=(crc,size)
print('official mic counts',collections.Counter(mic for uid,mic in audio),flush=True)
rows=[row for p in sorted(ROOT.glob('larry-*.json')) for row in json.loads(p.read_bytes())]
if len(rows)!=88156: raise ValueError(f'Incomplete mirror projection: {len(rows)}')
seen=set()
mismatch=collections.Counter()
grouped=collections.defaultdict(set)
for row in rows:
    uid,mic=row['id'],row['mic_id']
    k=uid,mic
    if k in seen: raise ValueError('Duplicate mirror recording')
    seen.add(k)
    speaker,prompt=uid.split('_')
    if speaker!=row['speaker_id'] or row['path']!=f'{speaker}/{uid}_{mic}.flac':
        mismatch['invalid_path_or_id']+=1
        continue
    if k not in audio or uid not in texts:
        mismatch['absent_official_pair']+=1
        continue
    if norm(texts[uid])!=norm(row['text_original']):
        mismatch['text']+=1
        continue
    duration = row['duration_s']
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid recording duration')
    if mic not in {'mic1', 'mic2'}:
        raise ValueError('Unrecognized microphone')
    if mic == 'mic1' and 1 <= duration <= 10 and norm(texts[uid]):
        grouped[(prompt,norm(texts[uid]))].add(speaker)
print('mirror counts',len(rows),collections.Counter(mic for uid,mic in seen),flush=True)
print('mismatch',mismatch,'unmirrored official',len(audio.keys()-seen),'unmirrored paired official',sum(uid in texts for uid,mic in audio.keys()-seen),flush=True)
if mismatch or any(uid in texts for uid, _ in audio.keys() - seen):
    raise ValueError('Official and mirror metadata differ for a recording with a transcript')
eligible=sorted((p,t,ss) for (p,t),ss in grouped.items() if len(ss)>=40)
print('prompts with >=40 speakers',[(p,len(ss)) for p,t,ss in eligible],flush=True)
first=sorted((p,t,ss) for p,t,ss in eligible if len(ss)>=64)[:24]
result=None
if len(first)==24 and len(set(p for p,t,ss in first))==24:
    common=sorted(set.intersection(*(ss for p,t,ss in first)))
    if len(common)>=64:
        result=(common[:64], [p for p,t,ss in first], 'first choice')

def find(s, p):
    def witness(start, common, labels):
        if len(labels)==p: return sorted(common)[:s],labels
        for i in range(start,len(eligible)-(p-len(labels))+1):
            pid, _, support = eligible[i]
            if pid in labels: continue
            both=support if common is None else common & support
            if len(both)>=s:
                found=witness(i+1,both,labels+[pid])
                if found is not None: return found
        return None
    return witness(0,None,[])
if result is None:
    for s,p in sorted(itertools.product(range(40,65),range(16,25)),key=lambda sp:(-sp[0]*sp[1],-sp[0],-sp[1])):
        answer=find(s,p)
        if answer:
            result=(*answer,'fallback witness')
            break
print('grid result',None if result is None else [len(result[0]),len(result[1]),result[2]],flush=True)
if result:
    b=json.dumps(dict(zip(('speakers','prompts','method'),result)),sort_keys=True).encode()
    write_bytes(ROOT / 'candidate-grid-audit.json', b)
    print('grid content digest',hashlib.sha256(b).hexdigest(),flush=True)
