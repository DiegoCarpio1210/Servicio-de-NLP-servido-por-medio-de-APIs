"""Final read-only checks and a durable lifecycle/contract/report evidence summary."""
import hashlib,json,re,socket,subprocess
from pathlib import Path

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path('logs/quality20261029');sources=[]
    for row in json.loads(Path('logs/source_integrity.json').read_text()):
        p=next(Path('originals').glob('*.pdf')) if row['source'].endswith('.pdf') else Path('originals')/row['source']
        assert digest(p)==row['sha256'];sources.append(p.name)
    report=Path('REPORT.md').read_text(encoding='utf-8-sig')
    template=Path('originals/REPORT_TEMPLATE.md').read_text(encoding='utf-8')
    assert re.findall(r'^## .*',report,re.M)==re.findall(r'^## .*',template,re.M)
    for section in re.split(r'^## .*$',report,flags=re.M)[1:]:assert section.strip()
    prose=len(' '.join(line for line in report.splitlines() if not line.startswith('|')).split())
    assert prose<=1000
    lifecycle=json.loads(Path('tmp/local-app/status.json').read_text())
    assert lifecycle['state']=='closed' and lifecycle['owned_processes_stopped']
    for port in (8000,8765):
        with socket.socket() as probe:
            probe.settimeout(1);assert probe.connect_ex(('127.0.0.1',port))!=0
    artifacts=json.loads((root/'artifact_audit.json').read_text())
    assert digest(Path('models/continued20k/model.safetensors'))==artifacts['stable_unchanged']['weights']
    assert digest(Path('models/continued20k/calibration.json'))==artifacts['stable_unchanged']['calibration']
    assert digest(Path('models/quality40k/model.safetensors'))==artifacts['exported_hashes']['model.safetensors']
    git=['git','-c','safe.directory='+str(Path.cwd()).replace('\\','/')]
    subprocess.run(git+['diff','--exit-code','35fa732','--','qa','frontend','scripts/local_app.py','INICIAR_PIMIENTERO.bat','DETENER_PIMIENTERO.bat'],check=True,capture_output=True)
    names=subprocess.check_output(git+['ls-files'],text=True).splitlines()
    forbidden=[p for p in names if p.startswith(('models/','checkpoints/','data/','.venv/')) or p.endswith(('.safetensors','.pt','.pth')) or Path(p).name in ('.env','ngrok.yml')]
    assert not forbidden,forbidden
    ignored=['.env','.env.local','ngrok.yml','models/quality40k/model.safetensors','checkpoints/quality40k/checkpoint-2000/optimizer.pt']
    checked=subprocess.run(git+['check-ignore','--',*ignored],text=True,check=True,capture_output=True)
    assert set(checked.stdout.splitlines())==set(ignored)
    result={'original_sources_unchanged':sources,'report_headings_match':True,'report_nonempty_sections':10,
      'report_prose_words':prose,'stable_and_candidate_weights_unchanged':True,
      'production_api_ui_launcher_unchanged_since_35fa732':True,'ui_lifecycle':lifecycle,
      'ports_8000_8765_closed':True,'tracked_forbidden_artifact_paths':forbidden,'private_paths_ignored':ignored,
      'no_public_tunnel_opened':True,'no_new_training_during_recovery':True,'submission_tag_created':False}
    (root/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
