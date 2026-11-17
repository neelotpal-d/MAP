"""Bounded, extrusion-conserving cleanup of absolute-XYZ / relative-E paths.

Only consecutive positive-E explicit XYZ G1 moves are simplified. Every comment,
travel, mode, feed change and other command is a barrier. Wipe XYZ trajectories
are retained exactly; their retraction is moved to a separate E-only command.
Start/end machine code is passed through byte-for-byte (text newline preserving).
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

TICKS = 100000
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
EXPLICIT = re.compile(r'^G1\s+X('+NUM+r')\s+Y('+NUM+r')\s+Z('+NUM+r')\s+E('+NUM+r')\s+F('+NUM+r')\s*$')
EW = re.compile(r'(?<!\S)E('+NUM+r')(?=\s|$)')
FW = re.compile(r'(?<!\S)F('+NUM+r')(?=\s|$)')
XYZ = re.compile(r'(?<!\S)[XYZ]'+NUM+r'(?=\s|$)')
AXIS = re.compile(r'(?<!\S)([XYZ])('+NUM+r')(?=\s|$)')
START = ';===== assembled layer path body start ====='
END = ';===== assembled layer path body end ====='


def ticks(token):
    value = float(token)
    result = round(value*TICKS)
    if not math.isfinite(value) or abs(result/TICKS-value)>1e-10:
        raise ValueError('Motion cleanup requires E precision of at most five decimal places.')
    return result


def eword(value):
    return f'E{value/TICKS:.5f}'


def clean_stream(incoming, outgoing, settings=None):
    cfg=dict(xyz_tolerance_mm=.005, extrusion_tolerance_mm=.00005,
             tiny_e_mm=.0001, max_span_mm=.5, max_chunk_moves=128)
    cfg.update(settings or {})
    cfg.pop('enabled',None)
    from travel_feed import TravelFeedLimiter
    travel_limiter=TravelFeedLimiter(cfg['travel_z_speed_limit']) if cfg.get('travel_z_speed_limit') is not None else None
    for key in ('xyz_tolerance_mm','extrusion_tolerance_mm','tiny_e_mm','max_span_mm'):
        if not math.isfinite(float(cfg[key])) or cfg[key]<=0: raise ValueError(f'Invalid {key}')
    if int(cfg['max_chunk_moves'])<2: raise ValueError('max_chunk_moves must be >=2')
    threshold=ticks(str(cfg['tiny_e_mm']))
    if threshold<1: raise ValueError('tiny_e_mm is below output E precision')
    stats=Counter(); started=time.perf_counter(); body=False; seen_start=seen_end=0
    position=[None]*3; chunk=[]; chunk_start=None; chunk_feed=None; arc=0.; wipe=None
    maximum_xyz_error=maximum_e_error=0.
    before_positive=after_positive=before_negative=after_negative=0

    def emit(line):
        if body and travel_limiter is not None:line=travel_limiter.line(line)
        outgoing.write(line)
        if not line.endswith(('\n','\r')): outgoing.write('\n')

    def flush():
        nonlocal chunk,chunk_start,chunk_feed,arc,maximum_xyz_error,maximum_e_error,after_positive
        if not chunk: return
        points=[chunk_start]+[r[1] for r in chunk]
        cumulative=[0]; lengths=[0.]
        max_density=0.
        for i,r in enumerate(chunk):
            cumulative.append(cumulative[-1]+r[2])
            distance=math.dist(points[i],points[i+1])
            lengths.append(lengths[-1]+distance)
            if distance>0: max_density=max(max_density,r[2]/distance)
        keep={0,len(chunk)}; pending=[(0,len(chunk))]
        while pending:
            a,b=pending.pop()
            if b-a<2: continue
            span=lengths[b]-lengths[a]
            if span<=0: keep.update(range(a,b+1));continue
            if math.dist(points[a],points[b])<1e-9:
                split=max(range(a+1,b),key=lambda i:math.dist(points[a],points[i]))
                keep.add(split);pending.extend(((a,split),(split,b)));continue
            worst=1.; split=None
            for i in range(a+1,b):
                t=(lengths[i]-lengths[a])/span
                predicted=tuple(x+t*(y-x) for x,y in zip(points[a],points[b]))
                dx=math.dist(points[i],predicted)
                de=abs(cumulative[i]-(cumulative[a]+t*(cumulative[b]-cumulative[a])))/TICKS
                ratio=max(dx/cfg['xyz_tolerance_mm'],de/cfg['extrusion_tolerance_mm'])
                if ratio>worst: worst=ratio;split=i
            if split is not None:
                keep.add(split);pending.extend(((a,split),(split,b)))
        keep=sorted(keep)
        amounts=[cumulative[b]-cumulative[a] for a,b in zip(keep,keep[1:])]
        # Carry only sub-threshold residuals inside this bounded, uninterrupted
        # deposition chunk. No E is moved across a role, jump, travel or wipe.
        allocated=[]; carry=0
        for value in amounts:
            carry+=value
            if carry>threshold: allocated.append(carry);carry=0
            else: allocated.append(0)
        nonzero=[i for i,v in enumerate(allocated) if v]
        isolated=not nonzero
        if isolated:
            # Entire isolated stroke contains <=tiny_e_mm filament. Preserve it
            # explicitly without a tiny E word on a moving command.
            emit('; motion cleanup: isolated residual extrusion')
            emit(f'G1 {eword(cumulative[-1])} F{chunk_feed:g}')
            after_positive+=cumulative[-1]
            stats['isolated_residual_primes']+=1
            stats['isolated_residual_E_ticks']+=cumulative[-1]
            carry=0
        elif carry:
            allocated[nonzero[-1]]+=carry
        assert sum(allocated)+(cumulative[-1] if isolated else 0)==cumulative[-1]
        emitted_e=cumulative[-1] if isolated else 0
        final_feed=chunk_feed
        for j,(a,b) in enumerate(zip(keep,keep[1:])):
            line=chunk[b-1][0]
            amount=allocated[j]
            if amount:
                line=EW.sub(eword(amount),line,count=1)
                distance=math.dist(points[a],points[b])
                # Do not create a higher nominal E/second than the highest
                # original edge in this same chunk when residuals move.
                safe_feed=min(chunk_feed,chunk_feed*max_density*distance/amount)
                if distance<=0 or safe_feed<=0: raise ValueError('Stationary depositing chord')
                if safe_feed<chunk_feed-1e-8:
                    line=FW.sub(f'F{max(.00001,math.floor(safe_feed*1e5)/1e5):.5f}',line,count=1)
                    stats['flow_limited_chords']+=1
                after_positive+=amount
            else:
                line=EW.sub('',line,count=1)
                stats['moving_tiny_E_words_removed']+=1
            # Independent per-chord bound at every original vertex. Linear
            # interpolation bounds the full original edges, not just vertices.
            span=lengths[b]-lengths[a]
            for i in range(a,b+1):
                t=0. if span==0 else (lengths[i]-lengths[a])/span
                predicted=tuple(x+t*(y-x) for x,y in zip(points[a],points[b]))
                maximum_xyz_error=max(maximum_xyz_error,math.dist(points[i],predicted))
                predicted_e=emitted_e+t*amount
                maximum_e_error=max(maximum_e_error,abs(cumulative[i]-predicted_e)/TICKS)
            emitted_e+=amount
            emit(line)
            final_feed=float(FW.search(line)[1])
        if final_feed!=chunk_feed:
            # Future partial-axis or E-only commands may inherit F. Restore
            # exactly the modal feed the unmodified input left at this barrier.
            emit(f'G1 F{chunk_feed:g}')
            stats['modal_feed_restores']+=1
        stats['positive_input_moves']+=len(chunk)
        stats['positive_output_moves']+=len(amounts)
        stats['chunks']+=1
        chunk=[];chunk_start=None;chunk_feed=None;arc=0.

    def finish_wipe():
        nonlocal wipe,before_negative,after_negative
        total=0;feed=None;rewritten=[]
        for line in wipe:
            code=line.partition(';')[0].strip()
            if code.startswith(('G0 ','G1 ')):
                e=EW.search(code)
                if e:
                    value=ticks(e[1])
                    if value>0: raise ValueError('Unexpected positive extrusion inside wipe')
                    if value and not XYZ.search(code): raise ValueError('Unexpected E-only wipe command')
                    total+=value
                    line=EW.sub('',line,count=1)
                f=FW.search(code)
                if f:
                    if feed is not None and feed!=float(f[1]): raise ValueError('Mixed wipe feedrates')
                    feed=float(f[1])
            rewritten.append(line)
        if total:
            if not feed or feed<=0: raise ValueError('Wipe feedrate unavailable')
            emit('; motion cleanup: retract before wipe (relative E)')
            emit(f'G1 {eword(total)} F{feed:g}')
            before_negative+=-total;after_negative+=-total
            stats['wipe_retractions_separated']+=1
        for line in rewritten: emit(line)
        wipe=None

    for index,line in enumerate(incoming,1):
        if index%500000==0:print(f'[motion cleanup] {index:,} input lines; {time.perf_counter()-started:.1f}s',flush=True)
        text=line.strip(); code=text.partition(';')[0].strip()
        if text==START:
            if body or seen_start: raise ValueError('Duplicate assembled body')
            body=True;seen_start+=1;emit(line);continue
        if text==END:
            flush()
            if wipe is not None: raise ValueError('Unclosed wipe')
            body=False;seen_end+=1;emit(line);continue
        if not body: emit(line);continue
        if code in ('M82','G91','G20') or code.startswith('G92'):
            raise ValueError('Motion cleanup requires absolute millimetre XYZ and relative E without body G92')
        if text=='; WIPE_START':
            flush();wipe=[line];continue
        if wipe is not None:
            wipe.append(line)
            if code.startswith(('G0 ','G1 ')):
                for a,v in AXIS.findall(code): position['XYZ'.index(a)]=float(v)
            if text=='; WIPE_END':finish_wipe()
            continue
        match=EXPLICIT.fullmatch(code)
        if match and float(match[4])>0 and all(v is not None for v in position):
            endpoint=tuple(float(match[i]) for i in (1,2,3)); value=ticks(match[4]);feed=float(match[5])
            distance=math.dist(position,endpoint)
            if distance>0:
                if chunk and (feed!=chunk_feed or arc+distance>cfg['max_span_mm'] or len(chunk)>=cfg['max_chunk_moves']):flush()
                if not chunk:chunk_start=tuple(position);chunk_feed=feed
                chunk.append((line,endpoint,value));arc+=distance
                before_positive+=value;position[:]=endpoint
                if len(chunk)>=cfg['max_chunk_moves']:flush()
                continue
        flush()
        if code.startswith(('G0 ','G1 ')):
            e=EW.search(code)
            if e:
                value=ticks(e[1])
                if value>0:before_positive+=value;after_positive+=value
                elif value<0:before_negative+=-value;after_negative+=-value
                if XYZ.search(code) and abs(value)<=threshold:
                    # Rare noncanonical/partial-axis moves: exact geometry is
                    # retained, and E is made explicit as a separate operation.
                    if value:
                        f=FW.search(code)
                        if not f:raise ValueError('Tiny-E fallback lacks explicit feedrate')
                        emit('; motion cleanup: separate residual extrusion')
                        emit(f'G1 {eword(value)} F{f[1]}')
                    line=EW.sub('',line,count=1)
                    stats['fallback_moving_E_words_removed']+=1
            for a,v in AXIS.findall(code):position['XYZ'.index(a)]=float(v)
        emit(line)
    flush()
    if seen_start!=1 or seen_end!=1 or wipe is not None:raise ValueError('Incomplete assembled body')
    assert before_positive==after_positive and before_negative==after_negative
    if maximum_xyz_error>cfg['xyz_tolerance_mm']+1e-9:raise AssertionError('Geometry error bound exceeded')
    # RDP E-profile tolerance plus at most one tiny-word quantum of carry.
    if maximum_e_error>cfg['extrusion_tolerance_mm']+cfg['tiny_e_mm']+1e-9:raise AssertionError('Extrusion profile error bound exceeded')
    return dict(settings=cfg,counts=dict(stats),positive_E_ticks_before=before_positive,positive_E_ticks_after=after_positive,
                negative_E_ticks_before=before_negative,negative_E_ticks_after=after_negative,
                maximum_xyz_error_mm=maximum_xyz_error,maximum_local_E_profile_error_mm=maximum_e_error,
                travel_Z_limited_moves=travel_limiter.limited_moves if travel_limiter else 0,
                travel_feed_restores=travel_limiter.feed_restores if travel_limiter else 0,
                seconds=time.perf_counter()-started)


def clean_file(source, destination, settings=None):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    with source.open(encoding='utf8',newline='') as incoming,destination.open('x',encoding='utf8',newline='') as outgoing:
        return clean_stream(incoming,outgoing,settings)


def clean_file_in_place(path,settings):
    path=Path(path);staged=path.with_name(path.name+'.cleanup_'+uuid4().hex)
    result=clean_file(path,staged,settings)
    os.replace(staged,path)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--report',required=True,type=Path)
    args=p.parse_args()
    result=clean_file(args.source,args.destination)
    args.report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
