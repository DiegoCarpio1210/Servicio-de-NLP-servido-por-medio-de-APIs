"""Read-only evidence audit; never creates a submission tag or fabricates results."""
import json
import re
import subprocess
from pathlib import Path


def main():
    checks=[]
    def record(name,passed,detail): checks.append({'requirement':name,'status':'PASSED' if passed else 'PENDING','detail':detail})
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
    record('Final calibrated model exists',Path('models/final/calibration.json').exists(),'Pilot does not satisfy the final-quality gate')
    record('Required team/repository/tunnel facts complete','PENDING' not in report,'Replace only with verified facts')
    record('External HTTPS evidence',Path('logs/latency-tunnel.json').exists(),'Run from another network and inspect results')
    Path('logs/audit.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    return 0 if all(c['status']=='PASSED' for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
