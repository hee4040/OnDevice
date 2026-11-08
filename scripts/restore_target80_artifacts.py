#!/usr/bin/env python3
"""Verify and restore the published numeric evidence; refuse to replace existing runs."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination-root',type=Path,default=ROOT)
    args=ap.parse_args();root=args.destination_root.resolve()
    package=ROOT/'baseline/reports/2026-11-08/target80/replay'
    manifest=json.loads((package/'manifest.json').read_text());archive=package/'numeric_artifacts.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:raise ValueError('Archive SHA-256 mismatch')
    destinations={'source':root/'results/20261107T091527.251697Z-adaptive-drop','experiment':root/'results/20261108-target80'}
    for path in destinations.values():
        if path.exists():raise FileExistsError(f'Preserving existing output: {path}; use another --destination-root')
    verified=[]
    with zipfile.ZipFile(archive) as source:
        expected={m['path'] for m in manifest['members']}
        if set(source.namelist())!=expected or len(source.namelist())!=len(expected):raise ValueError('Unexpected archive members')
        for member in manifest['members']:
            relative=PurePosixPath(member['path'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in destinations:raise ValueError('Unsafe member path')
            info=source.getinfo(member['path'])
            if info.file_size!=member['bytes'] or info.file_size>32*1024*1024:raise ValueError('Unexpected member size')
            data=source.read(member['path'])
            if hashlib.sha256(data).hexdigest()!=member['sha256']:raise ValueError('Member SHA-256 mismatch')
            verified.append((destinations[relative.parts[0]].joinpath(*relative.parts[1:]),data))
    for path,data in verified:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    # This path is environment metadata, not a numerical result. Preserve its original value.
    summary_path=destinations['experiment']/'gate_search/summary.json'
    summary=json.loads(summary_path.read_text());original_source=summary['source'];summary['source']=str(destinations['source'])
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    record={'archive_sha256':manifest['archive_sha256'],'verified_members':len(verified),
            'path_adjustment':{'file':'gate_search/summary.json','field':'source','original':original_source,'restored':summary['source']}}
    (destinations['experiment']/'restore_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Verified and restored',len(verified),'numeric evidence files.')
    for name,path in destinations.items():print(name+': '+str(path))

if __name__=='__main__':main()
