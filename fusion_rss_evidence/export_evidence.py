"""Read-only derived tables and portable isolation check of frozen outputs."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil,subprocess,sys,tempfile

ROOT=Path(__file__).resolve().parent
def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    raw=load(ROOT/'results.json.gz')
    write_csv(ROOT/'changed_actions.csv',raw['changed_actions'])
    with (ROOT/'predictions.csv').open(newline='') as f:pred=list(csv.DictReader(f))
    endpoint={(x['seed'],x['family'],x['k']):x for x in pred if float(x['tau'])==1}
    baseline={(x['seed'],x['family'],x['k']):x for x in pred if float(x['tau'])==0}
    origins={key for key,value in endpoint.items() if value['proposal']!=baseline[key]['proposal']}
    cases=[x for x in pred if (x['seed'],x['family'],x['k']) in origins]
    assert len(cases)==75,len(cases)
    write_csv(ROOT/'mechanism_cases.csv',cases)
    keep=('bridge_math.py','verify_cached.py','cached_states.json.gz',
          'cached_manifest.json','predictions.csv','summary.json')
    with tempfile.TemporaryDirectory(prefix='rss_bridge_isolated_') as td:
        isolated=Path(td)
        for name in keep:shutil.copyfile(ROOT/name,isolated/name)
        p=subprocess.run([sys.executable,str(isolated/'verify_cached.py')],cwd=isolated,
                         capture_output=True,text=True,env={'PYTHONPATH':str(isolated)})
        assert p.returncode==0,p.stderr
        check=json.loads(p.stdout)
        assert check['passed']
        check['isolated_directory_files']=list(keep)
        check['no_parent_engine_or_physical_data_copied']=True
        check['cached_input_sha256']=sha(ROOT/'cached_states.json.gz')
        dump(ROOT/'portable_verification.json',check)
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
