"""Local calibration of existing MAP CPU kernels; no settings are changed here.

Neelotpal Dutta. Additional author details may be added here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
STAGES=('layers','pipeline','distance')


def machine_info():
    info=dict(logical_cpus=os.cpu_count() or 1,platform=platform.system(),
        python=platform.python_version(),available_memory_bytes=None,cpu_percent=None,plugged_in=None)
    try:
        import psutil
        info.update(available_memory_bytes=int(psutil.virtual_memory().available),
            cpu_percent=float(psutil.cpu_percent(interval=.15)))
        battery=psutil.sensors_battery()
        if battery is not None:info['plugged_in']=bool(battery.power_plugged)
    except (ImportError,OSError,AttributeError,NotImplementedError):pass
    return info


def candidate_plan(maximum_processes=8,reserve_cpus=2,*,cpu_count=None,system=None,available_memory=None):
    for value in (maximum_processes,reserve_cpus):
        if isinstance(value,bool) or not isinstance(value,int):raise ValueError('Worker limits must be integers.')
    if not 1<=maximum_processes<=32 or reserve_cpus<0:raise ValueError('Choose 1..32 processes and a nonnegative CPU reserve.')
    cpu=max(1,cpu_count if cpu_count is not None else os.cpu_count() or 1)
    budget=max(1,cpu-reserve_cpus);limit=min(maximum_processes,budget)
    if available_memory is not None:
        # Keep half the available memory outside the benchmark, allowing a
        # conservative 512 MiB per process. This is a guard, not a RAM prediction.
        limit=min(limit,max(1,int(available_memory)//(1024*1024*1024)))
    counts=sorted({1,*[p for p in (2,4,8,12,16,24,32) if p<=limit],limit})
    windows=(system or platform.system())=='Windows'
    plan=[]
    for workers in counts:
        plan.append(dict(stage='layers',workers=workers,threads=1))
        # VTK remains serial on Windows, but the production exact support BVH
        # has its own independent native thread budget and still needs testing.
        if workers*2<=budget:plan.append(dict(stage='layers',workers=workers,threads=2))
        plan.append(dict(stage='pipeline',workers=workers,threads=1))
        if workers*2<=budget:plan.append(dict(stage='pipeline',workers=workers,threads=2))
    for threads in ([1] if windows else counts):plan.append(dict(stage='distance',workers=1,threads=threads))
    return plan


def recommend(trials,*,near_best=.05,maximum_variation=.20):
    """Only matching, repeatable successful trials can supply recommendations."""
    groups={}
    for row in trials:
        groups.setdefault((row['stage'],row['workers'],row['threads']),[]).append(row)
    summaries=[];recommendations={};warnings=[]
    for (stage,workers,threads),rows in groups.items():
        good=[r for r in rows if r.get('status')=='passed' and isinstance(r.get('wall_seconds'),(int,float))
            and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0]
        values=[r['wall_seconds'] for r in good]
        median=statistics.median(values) if values else None
        variation=(max(values)-min(values))/median if len(values)>1 else None
        signatures={r.get('signature') for r in good}
        valid_signature=all(isinstance(s,str) and len(s)==64 for s in signatures)
        stable=len(good)==len(rows) and len(good)>=2 and variation<=maximum_variation and len(signatures)==1 and valid_signature
        summaries.append(dict(stage=stage,workers=workers,threads=threads,samples=len(values),
            median_seconds=median,spread_fraction=variation,stable=stable,
            signature=next(iter(signatures)) if len(signatures)==1 else None))
    for stage in STAGES:
        options=[r for r in summaries if r['stage']==stage and r['stable']]
        # Numerical outputs must match across configurations, not just repeats.
        if len({r['signature'] for r in options})>1:
            warnings.append(stage+': output checks disagree; no recommendation.');continue
        if not options:
            warnings.append(stage+': no repeatable complete comparison; rerun with less background activity.');continue
        fastest=min(r['median_seconds'] for r in options)
        close=[r for r in options if r['median_seconds']<=fastest*(1+near_best)]
        selected=min(close,key=lambda r:(r['workers']*r['threads'],r['workers'],r['threads']))
        recommendations[stage]=dict(workers=selected['workers'],threads=selected['threads'],
            median_seconds=selected['median_seconds'],spread_fraction=selected['spread_fraction'])
    return summaries,recommendations,warnings


def write_html(path,report):
    rows=[]
    for row in report['comparisons']:
        seconds='failed' if row['median_seconds'] is None else f"{row['median_seconds']:.3f}"
        spread='—' if row['spread_fraction'] is None else f"{row['spread_fraction']:.0%}"
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in
            (row['stage'],row['workers'],row['threads'],row['samples'],seconds,spread,'yes' if row['stable'] else 'no'))+'</tr>')
    text='<!doctype html><meta charset="utf-8"><title>Worker calibration</title>'
    text+='<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
    text+='<h1>Worker calibration</h1><p>Measured isolated batch; timings include fresh process startup and output I/O. A smaller setting within 5% of the fastest stable result is preferred.</p>'
    text+='<h2>Recommendations</h2><pre>'+html.escape(json.dumps(report['recommendations'],indent=2))+'</pre>'
    text+='<p>'+html.escape(' '.join(report['warnings']))+'</p>'
    text+='<table><tr><th>Stage</th><th>Processes</th><th>Threads each</th><th>Runs</th><th>Median s</th><th>Spread</th><th>Stable</th></tr>'+''.join(rows)+'</table>'
    text+='<h2>Scope</h2><p>'+html.escape(report['scope'])+'</p>'
    path.write_text(text,encoding='utf8')


def run_calibration(output,*,mode='Quick',maximum_processes=8,reserve_cpus=2,repeats=2,timeout=180.,keep_artifacts=False):
    if mode not in ('Quick','Thorough') or isinstance(repeats,bool) or not isinstance(repeats,int) or not 2<=repeats<=5:
        raise ValueError('Choose Quick/Thorough and 2..5 repeats.')
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('Timeout must be positive.')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'calibration_job.json').write_text(json.dumps(dict(schema='map.worker-calibration.v1')),encoding='utf8')
    sources=[HERE/'calibration.py',HERE/'runtime/calibration_workload.py',
        HERE/'runtime/run_surface_pipeline_projected_manifest_support.py',
        HERE.parent/'CurvedSupportGenerator/runtime/SupportGenerationCurved/tree_geometry.py',
        HERE.parent/'CurvedSupportGenerator/runtime/SupportGenerationCurved/exact_support_bvh.py']
    def fingerprints():
        return {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_hashes=fingerprints()
    machine=machine_info();plan=candidate_plan(maximum_processes,reserve_cpus,cpu_count=machine['logical_cpus'],
        system=machine['platform'],available_memory=machine['available_memory_bytes'])
    samples=24 if mode=='Quick' else 64
    environment=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):environment[key]='1'
    worker=HERE/'runtime/calibration_workload.py'
    def execute(command,log):
        if __package__:
            from .process_tree import ProcessTree
        else:
            from process_tree import ProcessTree
        began=time.perf_counter()
        with log.open('w',encoding='utf8') as stream,subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
            env=environment,start_new_session=os.name!='nt',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0) as process:
            with ProcessTree(process) as tree:
                try:code=process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    tree.stop();process.wait();raise TimeoutError(f'Calibration trial exceeded {timeout:g} seconds; see {log}')
        return code,time.perf_counter()-began
    code,_=execute([sys.executable,'-B',str(worker),'prepare',str(root/'fixture'),'--samples',str(samples)],root/'prepare.log')
    if code:raise RuntimeError('Cannot prepare calibration fixture; see '+str(root/'prepare.log'))
    trials=[];total=len(plan)*repeats
    for repeat in range(repeats):
        order=plan.copy();random.Random(142+repeat).shuffle(order)
        for candidate in order:
            index=len(trials);name=f"{index:03d}_{candidate['stage']}_{candidate['workers']}x{candidate['threads']}"
            run=root/name;result=root/(name+'.json');log=root/(name+'.log')
            print(json.dumps(dict(calibration_progress=index,total=total,stage=candidate['stage'],workers=candidate['workers'],threads=candidate['threads'])),flush=True)
            trial=dict(candidate,repeat=repeat,status='failed',signature=None,wall_seconds=None,log=log.name)
            try:
                code,elapsed=execute([sys.executable,'-B',str(worker),candidate['stage'],str(root/'fixture'),
                    '--output',str(run),'--result',str(result),'--workers',str(candidate['workers']),'--threads',str(candidate['threads'])],log)
                trial['wall_seconds']=elapsed
                if code or not result.is_file():raise RuntimeError('Worker failed; see '+str(log))
                trial.update(json.loads(result.read_text(encoding='utf8')))
                # Rank full invocation wall time, not only an inner kernel timer.
                trial['wall_seconds']=elapsed
            except (RuntimeError,TimeoutError,OSError,ValueError) as error:trial['error']=str(error)
            trials.append(trial)
            (root/'trials.json').write_text(json.dumps(trials,indent=2),encoding='utf8')
            if run.is_dir() and not keep_artifacts:
                assert run.resolve().is_relative_to(root) and run.resolve()!=root
                shutil.rmtree(run)
    comparisons,recommendations,warnings=recommend(trials)
    if machine['plugged_in'] is False:warnings.append('Battery operation recorded; repeat on AC power before choosing a permanent profile.')
    if machine['cpu_percent'] is not None and machine['cpu_percent']>30:warnings.append('Background CPU activity was high at the start; repeat while idle.')
    if fingerprints()!=source_hashes:
        recommendations={};warnings.append('Source code changed during calibration; rerun before applying settings.')
    report=dict(schema='map.worker-calibration.v1',created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=source_hashes,
        machine=machine,mode=mode,samples=samples,repeats=repeats,trials=trials,comparisons=comparisons,
        recommendations=recommendations,warnings=warnings,can_apply=len(recommendations)==len(STAGES),
        scope='Synthetic representative batch using production CPU kernels. Layer extraction, complete projected model/support path + height pipeline, and routing distance-query kernels are measured. Routing search/merging, neural/GPU training, surface-field planning and other models are not certified or tuned. Windows VTK follows the existing serial policy. Filesystem cache is not flushed; each trial uses a fresh process. No existing job or settings are modified.')
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');write_html(root/'report.html',report)
    print(json.dumps(dict(calibration_progress=total,total=total,report=str(root/'report.json'),recommendations=recommendations)),flush=True)
    return root/'report.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=['Quick','Thorough'],default='Quick');p.add_argument('--maximum-processes',type=int,default=8)
    p.add_argument('--reserve-cpus',type=int,default=2);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--timeout',type=float,default=180.);p.add_argument('--keep-artifacts',action='store_true')
    a=p.parse_args();run_calibration(a.output,mode=a.mode,maximum_processes=a.maximum_processes,
        reserve_cpus=a.reserve_cpus,repeats=a.repeats,timeout=a.timeout,keep_artifacts=a.keep_artifacts)

if __name__=='__main__':main()
