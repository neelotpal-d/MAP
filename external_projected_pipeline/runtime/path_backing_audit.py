"""Chronological support-start screen and geometry-preserving seam relocation.

The box is a finite proximity screen, not a bonding/bridge certificate. Current
and future runs cannot witness a start. Empty runs and explicit travels do not
enter the deposition index. A compiler is optional: SciPy supplies a fallback.
"""
import json,time,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from adaptive_path_decimation import _load_tags,_write_tags,_load_break_indices


class SegmentIndex:
    def __init__(self,points,segments,ranks,parts,threads=2,backend='auto'):
        self.native=None;self.backend='scipy';self.reason=None
        self.ranks=np.asarray(ranks);self.parts=np.asarray(parts)
        if backend!='scipy':
            try:
                from deposition_bvh import TriangleAudit
                faces=segments[:,[0,1,1]]
                offsets=np.r_[0,np.flatnonzero(np.diff(ranks))+1,len(ranks)]
                self.native=TriangleAudit(points,faces,offsets,np.asarray(ranks)[offsets[:-1]],np.full(len(offsets)-1,3),
                    threads=threads,face_ranks=ranks)
                self.backend='native'
            except (OSError,RuntimeError,FileNotFoundError,TypeError,subprocess.SubprocessError) as exc:
                if backend=='native':raise
                self.reason=str(exc)
        if self.native is None:
            self.a=points[segments[:,0]];self.b=points[segments[:,1]]
            self.mid=(self.a+self.b)/2;half=np.linalg.norm(self.b-self.a,axis=1)/2
            # Length buckets prevent a single long bridge inflating every query.
            bucket=np.ceil(np.log2(np.maximum(half,.01))).astype(int)
            self.groups=[]
            for group in np.unique(bucket):
                ids=np.flatnonzero(bucket==group)
                self.groups.append((ids,cKDTree(self.mid[ids]),float(half[ids].max())))

    def close(self):
        if self.native:self.native.close()

    def query(self,points,ranks,lower,upper):
        q=np.asarray(points);ranks=np.asarray(ranks)
        if self.native:
            # A single geometry table uses mask 3; per-face chronology is exact.
            return self.native.query(q,ranks,3,lower,upper)[0]>=0
        lower=np.asarray(lower);upper=np.asarray(upper)
        centers=q+(lower+upper)/2;radius=np.linalg.norm((upper-lower)/2)
        hit=np.zeros(len(q),bool)
        for ids,tree,half in self.groups:
            active=np.flatnonzero(~hit)
            for i,local in zip(active,tree.query_ball_point(centers[active],radius+half)):
                candidates=ids[local];candidates=candidates[self.ranks[candidates]<ranks[i]]
                if not len(candidates):continue
                a=self.a[candidates];delta=self.b[candidates]-a
                lo=q[i]+lower;hi=q[i]+upper;moving=abs(delta)>1e-15
                parallel_outside=np.any((~moving)&((a<lo)|(a>hi)),axis=1)
                safe=np.where(moving,delta,1.)
                t0=(lo-a)/safe;t1=(hi-a)/safe
                enter=np.where(moving,np.minimum(t0,t1),-np.inf).max(axis=1)
                leave=np.where(moving,np.maximum(t0,t1),np.inf).min(axis=1)
                hit[i]=np.any(~parallel_outside&(np.maximum(enter,0)<=np.minimum(leave,1)))
        return hit


def load_records(folder,layers,*,positions_only=False,workers=1):
    def load(row):
        name=row.get('output_path_file') or row.get('path_file')
        if not name:raise ValueError('Missing scheduled path filename.')
        path=Path(folder)/name
        if path.stat().st_size==0:return None
        p=np.loadtxt(path,ndmin=2,usecols=(0,1,2) if positions_only else None)
        if p.shape[1]<3 or not np.isfinite(p).all():raise ValueError('Invalid path: '+str(path))
        breaks=_load_break_indices(Path(folder)/row['path_break_file'],len(p)) if row.get('path_break_file') else []
        # Export treats longer steps as travel even without a sidecar.
        breaks=sorted(set(breaks)|set((np.flatnonzero(np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1)>5.)+1).tolist()))
        return dict(row=row,p=p,breaks=breaks,path=path)
    ordered=sorted(layers,key=lambda r:(int(r['layer_main']),int(r['layer_sub'])))
    if workers<=1:return [r for r in map(load,ordered) if r is not None]
    with ThreadPoolExecutor(max_workers=min(4,int(workers))) as pool:
        return [r for r in pool.map(load,ordered) if r is not None]


def _index(records,threads=2,backend='auto'):
    vertices=[];segments=[];ranks=[];parts=[];runs=[];offset=0;rank=0
    for record in records:
        p=record['p'];vertices.append(p[:,:3]);bounds=np.r_[0,record['breaks'],len(p)]
        for lo,hi in zip(bounds[:-1],bounds[1:]):
            lo,hi=int(lo),int(hi);rank+=1
            active=np.flatnonzero(np.linalg.norm(np.diff(p[lo:hi,:3],axis=0),axis=1)>1e-10)+lo
            if not len(active):continue
            segments.append(np.column_stack((active+offset,active+offset+1)))
            ranks.append(np.full(len(active),rank,np.int32))
            parts.append(np.full(len(active),2 if record['row'].get('part')=='support' else 1,np.int8))
            runs.append(dict(record=record,lo=lo,hi=hi,rank=rank))
        offset+=len(p)
    index=SegmentIndex(np.vstack(vertices),np.concatenate(segments),np.concatenate(ranks),np.concatenate(parts),threads,backend) if segments else None
    return index,runs


def screen_records(records,*,width=.5,nominal=.2,bed=0.,threads=2,backend='auto'):
    if not np.isfinite([width,nominal,bed]).all() or min(width,nominal)<=0:raise ValueError('Invalid backing dimensions.')
    started=time.perf_counter();index,runs=_index(records,threads,backend)
    report=dict(schema='support-start-screen.v1',status='passed',findings=[],support_runs=0,queries=0,
        lower=[-1.2*width,-4*nominal,-1.2*width],upper=[1.2*width,.1*nominal,1.2*width],
        bed=bed,bed_start_tolerance=1.5*nominal,
        scope='Finite earlier-centreline proximity and supported-start screen. No bond strength, continuous backing, bridge calibration or rotating-tool certificate.')
    try:
        supports=[r for r in runs if r['record']['row'].get('part')=='support'];report['support_runs']=len(supports)
        if index is None:return report
        report['backend']=index.backend
        if index.reason:report['native_fallback_reason']=index.reason
        if not supports:return report
        starts=np.array([r['record']['p'][r['lo'],:3] for r in supports]);ranks=np.array([r['rank'] for r in supports],np.int32)
        hit=index.query(starts,ranks,report['lower'],report['upper']);report['queries']+=len(starts)
        for run,okay,start in zip(supports,hit,starts):
            if okay or bed-1e-8<=start[1]<=bed+report['bed_start_tolerance']:continue
            p=run['record']['p'][run['lo']:run['hi'],:3]
            sample=p[np.unique(np.linspace(0,len(p)-1,min(256,len(p))).astype(int))]
            anchors=index.query(sample,np.full(len(sample),run['rank'],np.int32),report['lower'],report['upper'])
            anchors|=(sample[:,1]>=bed-1e-8)&(sample[:,1]<=bed+report['bed_start_tolerance'])
            report['queries']+=len(sample);row=run['record']['row']
            report['findings'].append(dict(main=int(row['layer_main']),sub=int(row['layer_sub']),lo=run['lo'],hi=run['hi'],
                kind='unbacked_start' if anchors.any() else 'no_sampled_backing',point=start.tolist()))
        report['status']='failed' if report['findings'] else 'passed'
        return report
    finally:
        report['seconds']=time.perf_counter()-started
        if index:index.close()


def repair_support_seams(folder,layers,*,width=.5,nominal=.2,bed=0.,threads=2,backend='auto'):
    """Only reorder existing raw runs. Heights are calculated after this stage."""
    folder=Path(folder)
    print('[support backing] Reading XYZ once; normals/widths are loaded only for relocated seams.',flush=True)
    records=load_records(folder,layers,positions_only=True,workers=threads)
    before=screen_records(records,width=width,nominal=nominal,bed=bed,threads=threads,backend=backend)
    wanted={(f['main'],f['sub'],f['lo']) for f in before['findings']}
    changes=[]
    if wanted:
        index,runs=_index(records,threads,backend);pieces={id(r):[] for r in records}
        try:
            for run in runs:
                record=run['record'];row=record['row'];lo,hi=run['lo'],run['hi'];ids=np.arange(lo,hi)
                key=(int(row['layer_main']),int(row['layer_sub']),lo)
                if key in wanted:
                    p=record['p'][lo:hi,:3]
                    hits=index.query(p,np.full(len(p),run['rank'],np.int32),[-.9*width,-2.5*nominal,-.9*width],[.9*width,.1*nominal,.9*width])
                    hits|=(p[:,1]>=bed-1e-8)&(p[:,1]<=bed+1.5*nominal)
                    if hits[-1]:
                        ids=ids[::-1];action='reverse';selected=[ids]
                    elif hits.any():
                        pivot=int(np.flatnonzero(hits)[0]);selected=[ids[pivot:],ids[:pivot+1][::-1]];action='split_at_existing_anchor'
                    else:selected=[ids];action=None
                    if action:changes.append(dict(main=key[0],sub=key[1],lo=lo,action=action))
                else:selected=[ids]
                pieces[id(record)].extend(selected)
        finally:index.close()
        changed={(r['main'],r['sub']) for r in changes}
        for record in records:
            row=record['row'];key=(int(row['layer_main']),int(row['layer_sub']))
            if key not in changed:continue
            order=np.concatenate(pieces[id(record)]);bounds=np.cumsum([len(p) for p in pieces[id(record)]])[:-1]
            p=np.loadtxt(record['path'],ndmin=2)
            if p.shape[1]!=6:raise ValueError('Relocate seams before calculating heights, not on processed extrusion files.')
            tags=_load_tags(folder/row['path_tag_file'],len(p)) if row.get('path_tag_file') else None
            if tags is None or np.any(tags==''):raise ValueError('Seam repair requires complete role tags.')
            widths=np.loadtxt(folder/row['width_profile_file'],ndmin=1) if row.get('width_profile_file') else None
            if widths is None or len(widths)!=len(p):raise ValueError('Seam repair requires aligned widths.')
            np.savetxt(record['path'],p[order],fmt='%.10f');np.savetxt(folder/row['width_profile_file'],widths[order],fmt='%.10f')
            _write_tags(folder/row['path_tag_file'],tags[order]);stem=record['path'].stem
            break_file=folder/(stem+'.breaks.txt');break_file.write_text(''.join(f'{i}\n' for i in bounds))
            jump_file=folder/(stem+'.jumps.txt');length=np.linalg.norm(np.diff(p[order,:3],axis=0),axis=1)
            jumps=[(int(i),float(length[i-1])) for i in bounds if length[i-1]>2]
            jump_file.write_text(''.join(f'{i} {d:.10f} long_jump\n' for i,d in jumps))
            row.update(path_point_count=len(order),width_profile_point_count=len(order),path_break_file=break_file.name if len(bounds) else None,
                path_break_count=len(bounds),path_jump_file=jump_file.name if jumps else None,path_long_jump_count=len(jumps),
                path_tag_block_count=int(np.count_nonzero(np.r_[True,tags[order][1:]!=tags[order][:-1]])),support_seam_relocated=True)
            record.update(p=p[order,:3],breaks=bounds.tolist())
    # Screen the exact reordered geometry in memory; avoid rereading every file.
    after=screen_records(records,width=width,nominal=nominal,bed=bed,threads=threads,backend=backend)
    report=dict(before=before,changes=changes,after=after,status=after['status'],
        policy='Reverse or split/reorder existing segments; no added printed segment. Rebuild chronology and rescreen afterwards.')
    (folder/'support_backing.json').write_text(json.dumps(report,indent=2))
    return report
