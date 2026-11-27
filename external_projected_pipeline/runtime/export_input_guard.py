"""Freeze consumed export inputs and audit actual positive-E bare deposition."""
import argparse,hashlib,json,re,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from height_warning_policy import nominal_height,usable_heights
from path_backing_audit import screen_records


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _digests(paths):
    """Hash independent files concurrently; retain every byte and input order."""
    paths=list(paths)
    with ThreadPoolExecutor(max_workers=min(8,max(1,len(paths)))) as pool:
        return list(pool.map(digest,paths))


def _copy_frozen(item):
    path,out,sha=item
    out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    if digest(out)!=sha:raise ValueError('Input changed while copying: '+str(path))


def snapshot(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):raise ValueError('Private export snapshot must be disjoint from source.')
    metadata_path=source/'batch_3axis.meta.json';metadata_sha=digest(metadata_path)
    metadata=json.loads(metadata_path.read_text(encoding='utf8'));layers=metadata.get('layers',[])
    if not layers:raise ValueError('Cannot export an empty schedule.')
    consumed={metadata_path:metadata_sha};copies=[];keys=set();height_repairs={}
    raw=metadata.get('source_path_metadata',{});rawrows={(r['layer_main'],r['layer_sub']):r for r in raw.get('layers',[])}
    def bind(path,target):
        path=Path(path).resolve()
        if not path.is_file():raise FileNotFoundError('Missing export input: '+str(path))
        sha=digest(path)
        if path in consumed and consumed[path]!=sha:raise ValueError('Source changed while enumerating export inputs.')
        consumed[path]=sha;copies.append((path,target,sha));return target
    for row in layers:
        key=(int(row['layer_main']),int(row['layer_sub']));stem=f'{key[0]}_{key[1]}'
        if key in keys:raise ValueError('Duplicate layer identity in export schedule.')
        keys.add(key);upstream=rawrows.get(key,{})
        for name in ('surface_metric_mesh','surface_metric_components','surface_hatch_directions','parallel_axis'):
            if name not in row and name in upstream:row[name]=upstream[name]
        path=row.get('output_path_file') or (stem+'_3axis.txt')
        actual=source/path
        frozen_name=bind(source/path,stem+'_3axis.txt')
        row['path_point_count']=0;row['height_max']=0.
        if actual.stat().st_size:
            heights=np.loadtxt(actual,usecols=6,ndmin=1)
            bad=~np.isfinite(heights)|(heights<=0)
            if bad.any():
                heights=usable_heights(heights,nominal_height(metadata),label=str(actual))
                height_repairs[frozen_name]=heights
                row['export_height_fallback_points']=int(bad.sum())
            row['height_max']=float(heights.max(initial=0))
            row['path_point_count']=len(heights)
        row['output_path_file']=frozen_name
        for field,suffix in (('width_profile_file','.widths.txt'),('path_tag_file','.tags.txt'),
                ('path_break_file','.breaks.txt'),('path_jump_file','.jumps.txt'),('nominal_segment_length_file','.nominal_lengths.txt')):
            if row.get(field):row[field]=bind(source/row[field],stem+suffix)
        if row.get('width_profile_file'):
            source_width=next(p for p,target,_ in copies if target==row['width_profile_file'])
            profile=np.loadtxt(source_width,ndmin=1) if source_width.stat().st_size else np.empty(0)
            if profile.ndim!=1 or len(profile)!=row['path_point_count']:raise ValueError('Width profile must match the actual export path point count.')
            if not np.isfinite(profile).all() or (profile<=0).any():raise ValueError('Invalid actual export width profile.')
            row['width_profile_max']=float(profile.max(initial=0))
        from surface_metric_source import retained_metric_source
        mesh,recovery=retained_metric_source(row,metadata,base=source,upstream=upstream)
        if recovery:
            row['surface_metric_source_recovery']=recovery
            print(f'[export source] {stem}: restored retained layer geometry reference: {mesh}',flush=True)
        if mesh:
            target=bind(source/mesh,'surfaces/'+stem+'.obj')
            row['surface_metric_mesh']=str(destination/target)
        # Discard stale raw/patch-sidecar references. All consumed inputs are
        # materialized above in the current processed row.
        row.pop('connected_patches',None)
    metadata['source_path_metadata']={k:raw[k] for k in ('width','contour_spacing') if k in raw}
    metadata['output_dir']=str(destination)
    destination.mkdir(parents=True,exist_ok=False)
    if len({target for _,target,_ in copies})!=len(copies):raise ValueError('Duplicate private export target.')
    with ThreadPoolExecutor(max_workers=min(8,max(1,len(copies)))) as pool:
        list(pool.map(_copy_frozen,[(path,destination/target,sha) for path,target,sha in copies]))
    if any(actual!=expected for actual,expected in zip(_digests(consumed),consumed.values())):raise ValueError('Live export inputs changed during snapshot.')
    for name,heights in height_repairs.items():
        path=destination/name;rows=np.loadtxt(path,ndmin=2)
        if not np.isfinite(rows[:,:6]).all() or np.any(np.linalg.norm(rows[:,3:6],axis=1)<=1e-12):
            raise ValueError('Invalid path geometry/normals in export snapshot: '+str(path))
        rows[:,6]=heights
        np.savetxt(path,rows,fmt='%.10f')
    (destination/'batch_3axis.meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    files=[p for p in destination.rglob('*') if p.is_file()]
    report=dict(schema='export-snapshot.v1',source=str(source),source_files={str(p):sha for p,sha in consumed.items()},
        files={str(p.relative_to(destination)).replace('\\','/'):sha for p,sha in zip(files,_digests(files))})
    (destination/'snapshot.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def final_records(path,metadata):
    """Read the packaged program, including motion-fix E omissions and seams."""
    rows={(int(r['layer_main']),int(r['layer_sub'])):r for r in metadata['layers']}
    records=[];pieces=[];current=[];row=None;inside=False;seen=set();position=np.full(3,np.nan)
    def flush_run():
        nonlocal current
        if current:pieces.append(np.array(current));current=[]
    def flush_record():
        nonlocal pieces
        flush_run()
        if pieces:
            records.append(dict(row=row or dict(part='model',layer_main=-1,layer_sub=0),
                p=np.vstack(pieces),breaks=np.cumsum([len(p) for p in pieces])[:-1].tolist()))
            pieces=[]
    with Path(path).open(encoding='utf8') as source:
        for line in source:
            if line.startswith(';===== assembled layer path body start ====='):inside=True;continue
            if line.startswith(';===== assembled layer path body end ====='):flush_record();inside=False;break
            if not inside:continue
            if line.startswith('; ASSEMBLED_SNIPPET_START '):
                flush_record();m=re.search(r'\bmain=(\d+)\s+sub=(\d+)',line)
                if not m:raise ValueError('Missing packaged snippet identity.')
                key=tuple(map(int,m.groups()))
                if key not in rows:raise ValueError('Unknown packaged snippet identity.')
                row=rows[key];seen.add(key);continue
            if line.startswith('; ASSEMBLED_SNIPPET_END '):flush_record();row=None;continue
            code=line.split(';',1)[0].strip()
            if not re.match(r'^G[01](?:\s|$)',code):continue
            words=dict(_WORDS.findall(code));next_point=position.copy()
            for i,a in enumerate('XYZ'):
                if a in words:next_point[i]=float(words[a])
            moving=np.isfinite(position).all() and np.isfinite(next_point).all() and np.linalg.norm(next_point-position)>1e-10
            positive=moving and float(words.get('E',0))>0
            if positive:
                if not current:current=[position[[0,2,1]]*np.array([1,1,-1])]
                current.append(next_point[[0,2,1]]*np.array([1,1,-1]))
            elif moving:flush_run()
            position=next_point
    if inside:raise ValueError('Packaged assembled-body end marker missing.')
    if not seen and any(r.get('path_point_count',1)>1 for r in rows.values()):raise ValueError('No packaged snippets audited.')
    printed={(int(r['row']['layer_main']),int(r['row']['layer_sub'])) for r in records}
    missing=[key for key,row in rows.items() if row.get('part')=='support' and row.get('path_point_count',0)>1 and key not in printed]
    if missing:raise ValueError('Packaged program lost all positive-E deposition for support layers: '+str(missing[:10]))
    return records


def audit_final(path,snapshot_dir,report_path,*,check_support_backing=False):
    snapshot_dir=Path(snapshot_dir);verify_snapshot(snapshot_dir)
    meta=json.loads((snapshot_dir/'batch_3axis.meta.json').read_text())
    records=final_records(path,meta)
    report=screen_backing(records,enabled=check_support_backing,width=float(meta.get('width') or .5),
        nominal=float(meta.get('support_validation',{}).get('nominal_layer_height',.2)))
    report.update(gcode_sha256=digest(path),source_snapshot_sha256=digest(snapshot_dir/'snapshot.json'),
        actual_positive_E=True)
    Path(report_path).write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def verify_snapshot(folder):
    folder=Path(folder);report=json.loads((folder/'snapshot.json').read_text())
    actual=_digests(folder/name for name in report['files'])
    for (name,sha),value in zip(report['files'].items(),actual):
        if value!=sha:raise ValueError('Private export inputs changed: '+name)
    return report


_WORDS=re.compile(r'([XYZEF])\s*([-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?)')
def bare_records(folder,metadata,*,z_offset=0.,collect_collision=False,spacing_model_beads=False,surface_spacing_limit=None):
    """Bare format: explicit XYZ, relative E. Nonpositive E creates no bead."""
    folder=Path(folder);records=[];motion=[];move_orders=[];beads=[];bead_orders=[];radii=[];order=0
    from toolpath_order import ordered_layers
    for row in ordered_layers(metadata):
        stem=f"{row['layer_main']}_{row['layer_sub']}";path=folder/(stem+'.gcode')
        if not path.is_file():path=folder/(stem+'_0.gcode')
        if not path.is_file():raise FileNotFoundError('Missing bare snippet '+stem)
        previous=None;pieces=[];current=[]
        height=max(float(row.get('height_max',.5)),1e-6)
        width=float(row.get('width_profile_max') or metadata.get('width') or .5)
        if row.get('part')!='support' and spacing_model_beads:
            # Spacing-derived rounded-bead width exceeds spacing. Adding the
            # complete height is a conservative bound for both capsule branches.
            width=max(width,float(surface_spacing_limit or 0))+height
        with path.open(encoding='utf8') as stream:
            for line in stream:
                code=line.split(';',1)[0].strip()
                if not re.match(r'^G[01](?:\s|$)',code):continue
                words=dict(_WORDS.findall(code))
                if not all(k in words for k in 'XYZ'):raise ValueError('Bare audit requires explicit XYZ.')
                # Z-up export -> Y-up audit, a rigid proper rotation.
                point=np.array([float(words['X']),float(words['Z'])+z_offset,-float(words['Y'])])
                if not np.isfinite(point).all():raise ValueError('Nonfinite bare motion.')
                order+=1
                if previous is not None and collect_collision:
                    motion.append([previous,point]);move_orders.append(order)
                extruding=previous is not None and float(words.get('E',0))>0 and np.linalg.norm(point-previous)>1e-10
                if extruding:
                    if not current:current=[previous]
                    current.append(point)
                    if collect_collision:beads.append([previous,point]);bead_orders.append(order);radii.append(height+width/2)
                elif current:pieces.append(np.asarray(current));current=[]
                previous=point
        if current:pieces.append(np.asarray(current))
        if not pieces and row.get('part')=='support' and row.get('path_point_count',0)>1:
            raise ValueError('Scheduled support produced no positive-E deposition: '+stem)
        if pieces:
            records.append(dict(row=row,p=np.vstack(pieces),breaks=np.cumsum([len(p) for p in pieces])[:-1].tolist()))
    return records,(motion,move_orders,beads,bead_orders,radii)


def screen_backing(records,*,enabled,**kwargs):
    if not enabled:return dict(status='not_evaluated',check_enabled=False,reason='Unsupported support-run check disabled.')
    report=screen_records(records,**kwargs)
    report['check_enabled']=True
    if report['status']=='failed':
        report.update(status='warning',measured_status='failed')
        print(f"WARNING: {len(report['findings'])} unsupported support runs; continuing export.",flush=True)
    return report


def audit_bare(folder,snapshot_dir,report_path,*,tool_json=None,z_offset=0.,bed=0.,spacing_model_beads=False,surface_spacing_limit=None,check_support_backing=False):
    started=time.perf_counter();snapshot_dir=Path(snapshot_dir);verify_snapshot(snapshot_dir)
    meta=json.loads((snapshot_dir/'batch_3axis.meta.json').read_text())
    records,collision=bare_records(folder,meta,z_offset=z_offset,collect_collision=bool(tool_json),
        spacing_model_beads=spacing_model_beads,surface_spacing_limit=surface_spacing_limit)
    # Preserve source schedule identity for actual positive-E snippets. A failed
    # raw-path screen is only historical; this fresh screen governs publication.
    height=meta.get('support_validation',{});nominal=float(height.get('nominal_layer_height',.2))
    backing=screen_backing(records,enabled=check_support_backing,width=float(meta.get('width') or .5),nominal=nominal,bed=bed)
    result=dict(schema='export-deposition-screen.v1',status='warning' if backing['status']=='warning' else 'passed',backing=backing,
        source_snapshot_sha256=digest(snapshot_dir/'snapshot.json'),
        snippets={p.name:digest(p) for p in Path(folder).glob('*.gcode')},
        collision=dict(status='not_evaluated',reason='No fitted fixed-axis head preset supplied.'))
    if tool_json:
        from fixed_axis_collision import load_tool,audit
        lo,hi=load_tool(tool_json)
        result['collision']=audit(*collision,lo,hi)
        result['collision'].update(tool_file=str(Path(tool_json).resolve()),tool_sha256=digest(tool_json))
        if result['collision']['status']!='passed':result['status']='failed'
    result['seconds']=time.perf_counter()-started
    Path(report_path).write_text(json.dumps(result,indent=2),encoding='utf8')
    if result['status']=='failed':raise ValueError('Export deposition screen failed; inspect '+str(report_path))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('snapshot');s.add_argument('source',type=Path);s.add_argument('destination',type=Path)
    s=sub.add_parser('verify');s.add_argument('source',type=Path)
    s=sub.add_parser('audit-bare');s.add_argument('bare',type=Path);s.add_argument('source',type=Path);s.add_argument('report',type=Path)
    s.add_argument('--tool-json',type=Path);s.add_argument('--z-offset',type=float,default=0.)
    s.add_argument('--bed',type=float,default=0.)
    s.add_argument('--check-support-backing',action='store_true',help='Warn about unsupported support starts and continue.')
    s.add_argument('--spacing-model-beads',action='store_true');s.add_argument('--surface-spacing-limit',type=float)
    s=sub.add_parser('audit-final');s.add_argument('gcode',type=Path);s.add_argument('source',type=Path);s.add_argument('report',type=Path)
    s.add_argument('--check-support-backing',action='store_true',help='Warn about unsupported support starts and continue.')
    a=p.parse_args()
    if a.action=='snapshot':r=snapshot(a.source,a.destination)
    elif a.action=='verify':r=verify_snapshot(a.source)
    elif a.action=='audit-bare':r=audit_bare(a.bare,a.source,a.report,tool_json=a.tool_json,z_offset=a.z_offset,
        bed=a.bed,spacing_model_beads=a.spacing_model_beads,surface_spacing_limit=a.surface_spacing_limit,check_support_backing=a.check_support_backing)
    else:r=audit_final(a.gcode,a.source,a.report,check_support_backing=a.check_support_backing)
    print(json.dumps(dict(action=a.action,status=r.get('status','passed'),files=len(r.get('files',[])))),flush=True)


if __name__=='__main__':main()
