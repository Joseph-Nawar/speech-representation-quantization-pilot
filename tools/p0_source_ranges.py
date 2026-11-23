"""Fetch bounded metadata ranges from Edinburgh VCTK and a pinned mirror."""
import concurrent.futures
import hashlib
import io
import json
import re
import struct
import sys
import unicodedata
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

OFFICIAL='https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content'
MIRROR='https://huggingface.co/datasets/thelarryproject/vctk/resolve/0492fa00cf72afb16e753e95d0ec3439eebb07a6/train/train-{:05d}-of-00022.parquet'
ROOT=Path('cache/p0-source')
class Remote(io.RawIOBase):
    def __init__(self, url, total=None, etag=None):
        self.url=url
        self.total=total
        self.position=0
        self.bytes=0
        self.etag=etag
        self.allowed=None
        if total is None:
            self.pull(0,1)
    def pull(self, start, length):
        if length > 18_000_000 or self.bytes+length > 24_000_000:
            raise ValueError('Range budget exceeded')
        end=start+length-1
        suffix='&' if '?' in self.url else '?'
        with urlopen(Request(f'{self.url}{suffix}sq_range={start}-{end}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}),timeout=25) as response:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if response.status!=206 or not match or tuple(map(int,match.groups()[:2]))!=(start,end):
                raise ValueError(f'Invalid bounded range response {response.status} {response.headers.get("Content-Range")}')
            total=int(match[3])
            if self.total is not None and self.total!=total:
                raise ValueError('Remote length changed')
            self.total=total
            if self.etag and response.headers.get('ETag')!=self.etag:
                raise ValueError('Official ETag changed')
            payload=response.read(length+1)
        if len(payload)!=length:
            raise ValueError('Incomplete range')
        self.bytes+=length
        return payload
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position=offset+{0:0,1:self.position,2:self.total}[whence]
        return self.position
    def read(self,length=-1):
        length=min(self.total-self.position,length) if length>=0 else self.total-self.position
        if length<=0: return b''
        start=self.position
        self.position+=length
        if self.allowed is None:
            return self.pull(start,length)
        end=start+length
        output=bytearray(length)
        seen=False
        for lo,hi,data in self.allowed:
            left,right=max(lo,start),min(hi,end)
            if left<right:
                seen=True
                part=data[left-lo:right-lo] if data is not None else self.pull(left,right-left)
                output[left-start:right-start]=part
        if not seen:
            raise ValueError(f'Parquet requested unapproved range {start}:{end}')
        return bytes(output)

def write_bytes(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def get_mirror(index):
    import pyarrow.parquet as pq
    r=Remote(MIRROR.format(index))
    tail=r.pull(r.total-8,8)
    if tail[4:]!=b'PAR1': raise ValueError('Unrecognized Parquet footer')
    footer_length=struct.unpack('<I',tail[:4])[0]
    offset=r.total-8-footer_length
    footer=r.pull(offset,footer_length)+tail
    metadata=pq.read_metadata(io.BytesIO(b'PAR1'+footer))
    names={'id','speaker_id','mic_id','path','text_original','duration_s'}
    ranges=[]
    for group in range(metadata.num_row_groups):
        for col in range(metadata.num_columns):
            column=metadata.row_group(group).column(col)
            if column.path_in_schema in names:
                lo=column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
                ranges.append((lo,lo+column.total_compressed_size,None))
    r.allowed=[(offset,r.total,footer),*ranges]
    table=pq.ParquetFile(r,metadata=metadata,pre_buffer=False).read(columns=sorted(names),use_threads=False)
    path=ROOT/f'larry-{index:02}.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(table.to_pylist(),sort_keys=True))
    temporary.replace(path)
    return [index,len(table),r.bytes,hashlib.sha256(footer).hexdigest()]

def norm(t):
    return ' '.join(unicodedata.normalize('NFKC',t).casefold().replace('’',"'").split())

def official():
    r=Remote(OFFICIAL,11747302977,'"8a6ba2946b36fcbef0212cad601f4bfa"')
    with zipfile.ZipFile(r) as z:
        files=[(i.filename,i.CRC,i.file_size,i.compress_size,i.header_offset) for i in z.infolist()]
    b=json.dumps(files,separators=(',',':')).encode()
    write_bytes(ROOT / 'official-index.json', b)
    print('Official ZIP index',len(files),r.bytes,hashlib.sha256(b).hexdigest(),flush=True)
    print('Examples', [f[0] for f in files if re.search(r'/p225_00[12](?:_mic1\.flac|\.txt)$',f[0])],flush=True)

if __name__=='__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {"official", "mirror", "texts"}:
        raise SystemExit("usage: python tools/p0_source_ranges.py {official,mirror,texts}")
    ROOT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "official":
        official()
    if sys.argv[1]=='mirror':
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            pending={pool.submit(get_mirror,i):i for i in range(22)}
            records=[]
            for future in concurrent.futures.as_completed(pending):
                try:
                    result=future.result()
                    records.append(result)
                    print(result,flush=True)
                except Exception as error:
                    print('ERROR',pending[future],repr(error),flush=True)
            write_bytes(ROOT / 'mirror-ranges.json', json.dumps(sorted(records)).encode())
            if len(records)!=22: raise SystemExit(1)

if __name__=='__main__' and sys.argv[1]=='texts':
    r=Remote(OFFICIAL,11747302977,'"8a6ba2946b36fcbef0212cad601f4bfa"')
    with zipfile.ZipFile(r) as z:
        limit=min(i.header_offset for i in z.infolist() if i.filename.startswith('wav48_silence_trimmed/') and i.filename.endswith('.flac'))
        metadata=r.pull(0,limit)
        r.allowed=[(0,limit,metadata)]
        records={}
        for i in z.infolist():
            match=re.fullmatch(r'txt/([^/]+)/\1_(\d+)\.txt',i.filename)
            if match:
                k='_'.join(match.groups())
                if k in records: raise ValueError('Duplicate text entry')
                records[k]=z.read(i).decode('utf-8-sig')
    b=json.dumps(records,sort_keys=True).encode()
    write_bytes(ROOT / 'official-texts.json', b)
    print('Official texts',len(records),r.bytes,hashlib.sha256(metadata).hexdigest(),hashlib.sha256(b).hexdigest())
