"""Acquire the predeclared official collection only after the full source freeze."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, re, sys, time, urllib.request, zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('harth_authorized_acquisition',ROOT/'run_physical.py')
p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
def main():
    frozen=p.verify();protocol=p.load(ROOT/'protocol.json')
    url=protocol['proposed_archive_url']
    assert url=='https://archive.ics.uci.edu/static/public/779/harth.zip'
    target=ROOT/'raw/harth779.zip';target.parent.mkdir(exist_ok=True)
    assert not target.exists(), 'Existing acquisition must be verified, never silently replaced'
    part=target.with_suffix('.zip.part');assert not part.exists()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    digest=hashlib.sha256();count=0;last=0
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'Academic reproducibility verification'})
        with urllib.request.urlopen(request,timeout=60) as response,part.open('xb') as out:
            status=response.status;headers=dict(response.headers);resolved=response.geturl()
            assert status==200
            while True:
                chunk=response.read(1<<20)
                if not chunk:break
                count+=len(chunk);assert count<1_000_000_000,'Metadata-size guard exceeded'
                digest.update(chunk);out.write(chunk)
                if count-last>=10*(1<<20):
                    print('HARTH transfer MiB',round(count/(1<<20),1),'elapsed_s',round(time.monotonic()-clock,1),flush=True);last=count
        expected=headers.get('Content-Length')
        if expected is not None:assert count==int(expected)
        with zipfile.ZipFile(part) as archive:
            csvs=[name for name in archive.namelist() if re.fullmatch(r'S\d+\.csv',Path(name).name)]
            assert len(csvs)==22
        part.rename(target)
        record=dict(status='complete_official_acquisition_after_freeze',started_utc=started,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_url=url,resolved_url=resolved,
            bytes=count,archive_sha256=digest.hexdigest(),subject_members=csvs,http_headers=headers,
            algorithm_freeze_sha256=p.sha(ROOT/'algorithm_freeze.json'),protocol_sha256=p.sha(ROOT/'protocol.json'),
            authorization_sha256=p.sha(ROOT/'acquisition_authorization.json'),source_sha256=p.sha(__file__),
            interpretation='No physical outcomes read before algorithm,adapter,protocol and comparisons were frozen')
        p.dump(ROOT/'acquisition_record.json',record)
        print('HARTH acquisition complete',count,digest.hexdigest(),flush=True)
    except Exception as error:
        p.dump(ROOT/'acquisition_failure.json',dict(status='incomplete_no_task_evaluation',started_utc=started,
            error_type=type(error).__name__,error=str(error),bytes_received=count,source_sha256=p.sha(__file__)))
        raise
if __name__=='__main__':main()
