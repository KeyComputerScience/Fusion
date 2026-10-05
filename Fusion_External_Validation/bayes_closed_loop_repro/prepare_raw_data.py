"""Retrieve exact official UCI archives and safely extract the original inputs.

Network is needed only when --archives does not already contain name.zip.
Archive SHA256 must match the original experiment. Nested HAR ZIP is checked
separately. No scientific executable or supplied result is edited.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,stat,tempfile,urllib.request,zipfile
from pathlib import Path,PurePosixPath,PureWindowsPath

ROOT=Path(__file__).resolve().parent
SOURCES={
 'occupancy357':dict(url='https://archive.ics.uci.edu/static/public/357/occupancy%2Bdetection.zip',sha256='4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a',bytes=335713),
 'occupancy864':dict(url='https://archive.ics.uci.edu/static/public/864/room%2Boccupancy%2Bestimation.zip',sha256='a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd',bytes=114333),
 'mhealth319':dict(url='https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip',sha256='16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0',bytes=75567983),
 'har240':dict(url='https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip',sha256='c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031',bytes=61005872,nested_sha256='2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def verified_archive(name,folder):
    spec=SOURCES[name];path=folder/(name+'.zip')
    if not path.exists():
        folder.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.zip.partial')
        request=urllib.request.Request(spec['url'],headers={'User-Agent':'BayesianFusionReproduction/1.0'})
        try:
            with urllib.request.urlopen(request,timeout=120) as source,temp.open('wb') as dest:
                shutil.copyfileobj(source,dest,1<<20)
            if sha(temp)!=spec['sha256']:raise ValueError('Downloaded archive checksum differs: '+name)
            temp.replace(path)
        finally:
            if temp.exists():temp.unlink()
    if sha(path)!=spec['sha256'] or path.stat().st_size!=spec['bytes']:
        raise ValueError('Exact original archive required: '+str(path))
    return path


def validated_members(z):
    total=0
    for member in z.infolist():
        normalized=member.filename.replace('\\','/');p=PurePosixPath(normalized)
        if p.is_absolute() or '..' in p.parts or PureWindowsPath(normalized).drive:
            raise ValueError('Unsafe ZIP path: '+member.filename)
        mode=member.external_attr>>16
        if stat.S_ISLNK(mode):raise ValueError('ZIP symlink rejected: '+member.filename)
        total+=member.file_size
        if total>2*(1<<30):raise ValueError('ZIP expands beyond declared safety limit')
        yield member,p


def safe_extract(path,destination):
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        entries=list(validated_members(z))
        for member,relative in entries:
            target=destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):raise ValueError('ZIP path escapes destination')
            if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            # Exact source data may replace old extracted copies; cached arrays and
            # scientific code have different names and are left untouched.
            with z.open(member) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest,1<<20)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'independent_data');parser.add_argument('--archives',type=Path,default=ROOT/'raw_archives');parser.add_argument('--datasets',nargs='+',default=list(SOURCES),choices=list(SOURCES));parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();report={}
    for name in args.datasets:
        archive=verified_archive(name,args.archives)
        with zipfile.ZipFile(archive) as z:list(validated_members(z))
        if not args.verify_only:
            folder=args.data/name;safe_extract(archive,folder)
            if name=='har240':
                nested=folder/'UCI HAR Dataset.zip'
                if sha(nested)!=SOURCES[name]['nested_sha256']:raise ValueError('Nested HAR archive checksum differs')
                safe_extract(nested,folder)
        report[name]=dict(**SOURCES[name],verified=True,extracted=not args.verify_only)
        print('RAW_DATA_VERIFIED',name,flush=True)
    (args.archives/'raw_download_verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
