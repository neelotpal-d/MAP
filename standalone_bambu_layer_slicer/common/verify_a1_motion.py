"""Stream an assembled A1 body before packaging; no GUI or geometry dependency."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

START = ';===== assembled layer path body start ====='
END = ';===== assembled layer path body end ====='
WORDS = re.compile(r'([XYZEF])\s*([^\s]+)')
FILE = re.compile(r'\bfile=(\S+)')


def verify_a1_motion(path, *, maximum_z_speed=20., maximum_stationary_extrusion=1.,
                     tiny_e=1e-4, restart_tolerance=.01):
    """Check modes, pressure recovery, small moving E and clearance approaches.

    This checks commanded motion. It does not simulate firmware, acceleration,
    pressure or continuous material/head clearance. Ending withdrawal is reported
    separately; a withdrawal is required to be repaid before later deposition.
    """
    if not all(math.isfinite(v) and v > 0 for v in
               (maximum_z_speed, maximum_stationary_extrusion, tiny_e, restart_tolerance)):
        raise ValueError('A1 verification limits must be finite and positive.')
    started=time.perf_counter(); body=False; starts=ends=moves=deposits=approaches=0
    position=[None]*3; feed=None; debt=net=0.; active=None; snippet=''
    errors=[]; error_count=0; maximum_prime=stationary_run=max_run=max_run_seconds=run_seconds=0.
    max_z_speed=0.; missing_restarts=tiny_moves=negative_moves=0
    def fail(number, reason, **details):
        nonlocal error_count
        error_count+=1
        if len(errors)<100:errors.append(dict(line=number,reason=reason,snippet=snippet,**details))
    with Path(path).open(encoding='utf8') as stream:
        for number,raw in enumerate(stream,1):
            line=raw.strip()
            if line==START:
                starts+=1;body=True
                if starts>1:fail(number,'multiple assembled bodies')
                continue
            if line==END:
                ends+=1;body=False
                if active:fail(number,'unterminated approach')
                continue
            if not body:continue
            code=line.partition(';')[0].strip()
            if line.startswith('; ASSEMBLED_SNIPPET_START '):
                match=FILE.search(line);snippet=match[1] if match else line
                if active:
                    if not active['xy'] or not active['z']:fail(number,'approach missing XY or descent')
                    if snippet!=active['file']:fail(number,'approach assigned to different snippet')
                    approaches+=1;active=None
            elif line.startswith('; ASSEMBLED_APPROACH_SKIPPED'):
                fail(number,'skipped clearance approach')
            elif line.startswith('; ASSEMBLED_APPROACH reason='):
                if active:fail(number,'nested approach')
                match=FILE.search(line)
                if match:active=dict(file=match[1],xy=False,z=False,primes=0)
                else:fail(number,'approach has no source filename')
            if code in ('M82','G91','G20') or code.startswith('G92'):
                fail(number,'unsupported body coordinate/extrusion mode',command=code)
            if not code.startswith(('G0 ','G1 ')):continue
            moves+=1
            try:values={key:float(value) for key,value in WORDS.findall(code)}
            except ValueError:
                fail(number,'invalid motion number',command=code);continue
            if not all(math.isfinite(v) for v in values.values()):
                fail(number,'nonfinite motion number',command=code);continue
            old=position.copy()
            for i,axis in enumerate('XYZ'):
                if axis in values:position[i]=values[axis]
            feed=values.get('F',feed)
            if feed is not None and feed<=0:fail(number,'nonpositive feedrate')
            if any(v is not None and (v < -1e-6 or v > 256.000001) for v in position):
                fail(number,'motion outside A1 bounds',position=position.copy())
            moved=position!=old and None not in old+position
            e=values.get('E',0.);net+=e
            if moved:
                stationary_run=run_seconds=0.
                if 'E' in values and abs(e)<=tiny_e+1e-12:
                    tiny_moves+=1;fail(number,'tiny or zero moving E',E=e)
                if e<0:
                    negative_moves+=1;fail(number,'negative moving E',E=e)
                if e<=0 and feed is not None and feed>0:
                    length=math.dist(old,position)
                    z_speed=feed/60*abs(position[2]-old[2])/length
                    max_z_speed=max(max_z_speed,z_speed)
                    if z_speed>maximum_z_speed+1e-6:fail(number,'Z travel speed exceeds configured limit',z_speed=z_speed)
            elif e>0:
                maximum_prime=max(maximum_prime,e);stationary_run+=e
                if feed is not None and feed>0:run_seconds+=e*60/feed
                max_run=max(max_run,stationary_run);max_run_seconds=max(max_run_seconds,run_seconds)
                if stationary_run>maximum_stationary_extrusion+1e-8:
                    fail(number,'stationary extrusion exceeds configured limit',extrusion_mm=stationary_run)
            if e<0:debt-=e
            elif e>0 and not moved:debt=max(0.,debt-e)
            elif e>0:
                deposits+=1
                if debt>restart_tolerance:
                    missing_restarts+=1;fail(number,'withdrawal not recovered before deposition',debt_mm=debt)
                debt=0.
            if active:
                if 'X' in values or 'Y' in values:
                    if active['z'] or active['primes']:fail(number,'approach XY after descent or prime')
                    active['xy']=True
                if 'Z' in values:
                    if not active['xy'] or active['primes']:fail(number,'approach descent before XY or after prime')
                    active['z']=True
                if e>0:
                    if not active['z']:fail(number,'approach prime before descent')
                    active['primes']+=1
    if starts!=1 or ends!=1:fail(0,'expected exactly one complete assembled body',starts=starts,ends=ends)
    if active:fail(0,'unterminated approach')
    return dict(status='passed' if not error_count else 'failed',input=str(Path(path).resolve()),
        error_count=error_count,errors=errors,motion_commands=moves,positive_print_moves=deposits,
        approaches_checked=approaches,failed_restarts=missing_restarts,ending_debt_mm=debt,
        moving_tiny_E=tiny_moves,moving_negative_E=negative_moves,body_net_E_mm=net,
        maximum_stationary_prime_mm=maximum_prime,maximum_stationary_extrusion_run_mm=max_run,
        maximum_stationary_extrusion_run_requested_seconds=max_run_seconds,
        maximum_Z_travel_speed_mm_s=max_z_speed,seconds=time.perf_counter()-started,
        limits=dict(maximum_z_speed=maximum_z_speed,maximum_stationary_extrusion=maximum_stationary_extrusion,
                    tiny_e=tiny_e,restart_tolerance=restart_tolerance),
        scope='Assembled body commands only; no firmware/pressure simulation or geometric collision certificate.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('gcode',type=Path);p.add_argument('report',type=Path)
    p.add_argument('--maximum-z-speed',type=float,default=20.);a=p.parse_args()
    report=verify_a1_motion(a.gcode,maximum_z_speed=a.maximum_z_speed)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)
    if report['status']!='passed':raise SystemExit(f'A1 motion verification failed; inspect {a.report}')


if __name__=='__main__':main()
