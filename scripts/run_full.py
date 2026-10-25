"""Long-running authorized full-data job; each stage records its genuine outcome."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

root=Path('logs/final'); root.mkdir(parents=True,exist_ok=True)
status={'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'stages':[]}
def save(): (root/'job.json').write_text(json.dumps(status,indent=2))
commands=[('training',[sys.executable,'-m','scripts.train','--config','configs/train.json']),
          ('calibration',[sys.executable,'-m','scripts.evaluate','--model-dir','models/final']),
          ('http',[sys.executable,'-m','scripts.verify_local','--model-dir','models/final','--repeat','10'])]
save()
for name,command in commands:
    status['current_stage']=name; save()
    with (root/(name+'_execution.txt')).open('w') as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
        status['child_pid']=process.pid; save()
        code=process.wait()
    status['stages'].append({'name':name,'returncode':code}); save()
    if code:
        status['state']='failed' if name!='http' else 'quality_checks_failed'
        save(); raise SystemExit(code)
status['state']='pipeline_finished_requires_report_and_external_review'; save()
