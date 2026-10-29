"""Read-only evidence audit; never creates a submission tag or fabricates results."""
import json
import re
import subprocess
from pathlib import Path


def main():
    checks=[]
    def record(name,passed,detail,failure_status='PENDING'): checks.append({'requirement':name,'status':'PASSED' if passed else failure_status,'detail':detail})
    expected=re.findall(r'^## [0-9]\..*$',Path('originals/REPORT_TEMPLATE.md').read_text(encoding='utf-8'),re.M)
    report=Path('REPORT.md').read_text(encoding='utf-8')
    headings=re.findall(r'^## [0-9]\..*$',report,re.M)
    record('Report headings 0-9 unchanged and ordered',headings==expected,str(headings))
    sections=re.split(r'^## [0-9]\..*$',report,flags=re.M)[1:]
    record('Every report section populated',len(sections)==10 and all(s.strip() for s in sections),'Draft content is not final evidence')
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    forbidden=[f for f in tracked if f.endswith(('.safetensors','.bin','.pt','.pth')) or f=='.env' or f.startswith(('models/','checkpoints/','.venv/','data/','cache/'))]
    record('No tracked weights, private env, cache or dataset',not forbidden,str(forbidden))
    suspicious=[]
    pattern=re.compile(r'QA_API_TOKEN\s*=\s*[\"\']([^\"\']+)')
    for name in tracked:
        p=Path(name)
        if p.suffix in ('.py','.ps1','.md','.txt','.json','.yaml','.example'):
            for value in pattern.findall(p.read_text(encoding='utf-8',errors='replace')):
                if not value.startswith('<'): suspicious.append(name)
    record('No literal API token assignments in tracked text',not suspicious,str(suspicious))
    count=int(subprocess.check_output(['git','rev-list','--count','HEAD'],text=True))
    record('Multiple authentic commits',count>=2,str(count))
    candidate=json.loads(Path('configs/candidate.json').read_text()) if Path('configs/candidate.json').exists() else {}
    record('Calibrated current candidate exists',bool(candidate) and (Path(candidate['model_dir'])/'calibration.json').exists(),candidate.get('model_dir','No candidate'))
    record('Optional engineering quality target',candidate.get('decision')=='provisional_candidate','Independent-validation heuristic, not a mandatory professor cutoff','FAILED')
    record('Final submission fields complete','PENDING' not in report,'Submission tag and instructor dry-run URL intentionally pending')
    tunnel=Path('logs')/Path(candidate.get('model_dir','models/subset12k')).name/'tunnel-summary.json'
    import hashlib
    tunnel_evidence=json.loads(tunnel.read_text()) if tunnel.exists() else {}
    model_path=Path(candidate.get('model_dir','models/subset12k'))
    digest=None
    if (model_path/'model.safetensors').exists():
        with (model_path/'model.safetensors').open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    calibration=json.loads((model_path/'calibration.json').read_text()) if (model_path/'calibration.json').exists() else {}
    current_tunnel=(tunnel_evidence.get('timeout_budget_pass',False) and
                    tunnel_evidence.get('checkpoint_sha256')==digest and
                    tunnel_evidence.get('threshold')==calibration.get('threshold'))
    record('Current candidate and threshold HTTPS round trip',current_tunnel,
           'Requires matching weight hash and threshold; HTTPS from this host is separate from another-network rehearsal')
    external_path=Path('logs/external_network_verification.json')
    external=json.loads(external_path.read_text()) if external_path.exists() else {}
    external_pass=(external.get('technical_pass') is True and external.get('preflight_pass') is True and
                   external.get('https') is True and external.get('completed_requests')==50 and
                   external.get('network_scope')=='operator declares another network' and
                   external.get('info',{}).get('checkpoint_sha256')==digest and
                   external.get('info',{}).get('no_answer_threshold')==calibration.get('threshold'))
    mobile_path=Path('logs/continued20k/mobile_connectivity.json')
    mobile=json.loads(mobile_path.read_text(encoding='utf-8')) if mobile_path.exists() else {}
    mobile_checks=mobile.get('checks',[])
    mobile_pass=(len(mobile_checks)==5 and all(c.get('result')=='passed' for c in mobile_checks)
                 and mobile.get('info',{}).get('checkpoint_sha256')==digest
                 and mobile.get('info',{}).get('no_answer_threshold')==calibration.get('threshold')
                 and mobile.get('closure',{}).get('public_api_unavailable') is True)
    record('Five manual independent-network functional checks',mobile_pass,
           'User-reported Samsung mobile data checks corroborated by server status logs; no mobile timing claim')
    record('Independent-network 50-request timing rehearsal',external_pass,
           'Requires successful current-model HTTPS evidence and explicit operator network declaration; file existence alone is insufficient')
    Path('logs/audit.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    return 0 if all(c['status']=='PASSED' for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
