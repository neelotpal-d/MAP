"""MAP toolpath scheduling, independent of any machine exporter.

Neelotpal Dutta; additional author details: ______________________________.
Only complete patches at the same scalar/part/main level may move. File and
geometry identities remain stable; consumers follow the saved execution rank.
"""
import json
import math
from pathlib import Path
import time

SCHEMA = 'map.toolpath-order.v1'


def identity(row):
    return int(row['layer_main']), int(row['layer_sub'])


def ordered_layers(metadata):
    """Read a complete saved order, or the legacy main/sub order. Never replan."""
    rows = metadata.get('layers', []) if isinstance(metadata, dict) else metadata
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate processed layer identity in toolpath schedule.')
    present = ['toolpath_order' in r for r in rows]
    if any(present):
        values = [r.get('toolpath_order') for r in rows]
        if not all(present) or any(type(v) is not int for v in values) or sorted(values) != list(range(len(rows))):
            raise ValueError('Incomplete or duplicate saved toolpath execution ranks.')
        ordered=sorted(rows, key=lambda r:r['toolpath_order'])
        base=sorted(rows,key=identity);group_ids={};previous=None;group=-1
        for r in base:
            key=(int(r['layer_main']),r.get('part','model'),r.get('scalar'),r.get('material_index',0),
                 identity(r) if r.get('is_planar_base_layer') else None)
            if key!=previous:group+=1;previous=key
            group_ids[identity(r)]=group
        actual=[group_ids[identity(r)] for r in ordered]
        if actual!=sorted(actual):raise ValueError('Saved toolpath order crosses a scalar, part, main or base dependency.')
        return ordered
    return sorted(rows, key=identity)


def nearby_order(groups, endpoints, locked=None):
    """Endpoint heuristic with incoming/outgoing cost; never reverse a patch."""
    locked = locked or [False]*len(groups)
    result=[];previous=None;start=0;changed=eligible=0
    while start<len(groups):
        end=start+1
        while end<len(groups) and groups[end]==groups[start]:end+=1
        original=list(range(start,end));chosen=original
        if len(original)>1 and not any(locked[i] for i in original):
            eligible+=1;remaining=original.copy();candidate=[];current=previous
            if current is None:
                candidate.append(remaining.pop(0));current=endpoints[candidate[-1]][1]
            while remaining:
                i=min(remaining,key=lambda j:(math.dist(current,endpoints[j][0]),j))
                candidate.append(i);remaining.remove(i);current=endpoints[i][1]
            following=endpoints[end][0] if end<len(groups) else None
            def cost(indices):
                total=0.;current=previous
                for i in indices:
                    if current is not None:total+=math.dist(current,endpoints[i][0])
                    current=endpoints[i][1]
                if following is not None:total+=math.dist(current,following)
                return total
            if cost(candidate)<cost(original)-1e-9:chosen=candidate;changed+=1
        result.extend(chosen);previous=endpoints[chosen[-1]][1];start=end
    return result,dict(eligible_groups=eligible,reordered_groups=changed,
                       patches_moved=sum(i!=j for i,j in enumerate(result)))


def _ends(path):
    """Read only first/last numeric rows, independent of the number of waypoints."""
    with path.open('rb') as stream:
        first=stream.readline()
        if not first.strip():return None
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-4096))
        tail=stream.read().splitlines();last=next((r for r in reversed(tail) if r.strip()),first)
    values=tuple(tuple(float(v) for v in line.split()[:3]) for line in (first,last))
    if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in values):
        raise ValueError('Invalid toolpath endpoint: '+str(path))
    return values


def plan_layers(folder, metadata, enabled=True):
    rows=sorted(metadata['layers'],key=identity)
    groups=[];endpoints=[];locked=[]
    for i,row in enumerate(rows):
        name=row.get('output_path_file') or row.get('path_file')
        ends=_ends(Path(folder)/name) if name else None
        value=row.get('scalar');part=row.get('part','model')
        if value is not None and not math.isfinite(float(value)):raise ValueError('Nonfinite layer scalar.')
        # Model-only main/sub filenames are one scalar layer and its patches.
        group=(int(row['layer_main']),part,float(value) if value is not None else None,row.get('material_index',0))
        roles=list(row.get('path_tag_values',[]))
        tag=row.get('path_tag_file')
        if tag and (Path(folder)/tag).is_file():roles.extend((Path(folder)/tag).read_text().splitlines())
        repair=any('repair' in str(r) or 'hatch' in str(r) for r in roles)
        repair=repair or bool(row.get('support_reference_additions')) or bool(row.get('anchored_hole_height',{}).get('qualified_points',0))
        lock=bool(row.get('is_planar_base_layer')) or ends is None or repair or not enabled
        groups.append(('locked',i) if row.get('is_planar_base_layer') or ends is None else group)
        endpoints.append(ends or ((0.,0.,0.),(0.,0.,0.)));locked.append(lock)
    order,report=nearby_order(groups,endpoints,locked)
    planned=[dict(rows[i],toolpath_order=rank) for rank,i in enumerate(order)]
    report.update(schema=SCHEMA,enabled=bool(enabled),method='equal-scalar patch endpoint proximity',
                  internal_paths_preserved=True,repair_patches_locked=sum(locked),
                  sequence=[list(identity(r)) for r in planned])
    return planned,report


def _write(path,data):
    temporary=path.with_name(path.name+'.order.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8');temporary.replace(path)


def finalize_job(config_path):
    """Standard pipeline finalization, before any viewer or machine export."""
    config_path=Path(config_path);config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    paths={k:(config_path.parent/str(v)).resolve() for k,v in config['paths'].items()}
    from draft_toolpaths import enabled, METADATA
    if enabled(config):
        folder=paths['toolpaths']
        metadata=json.loads((folder/METADATA).read_text(encoding='utf8'))
        layers,report=plan_layers(folder,metadata,config.get('toolpath_ordering',{}).get('enabled',True))
        report.update(support_backing_check_enabled=False,support_backing_status='not_evaluated')
        metadata.update(layers=layers,toolpath_ordering=report)
        for name in (METADATA,'batch_paths.meta.json'):_write(folder/name,metadata)
        _write(folder/'toolpath_order.json',dict(schema=SCHEMA,layers=[dict(layer_main=r['layer_main'],
            layer_sub=r['layer_sub'],toolpath_order=r['toolpath_order']) for r in layers]))
        return report
    folder=paths['heights'];raw_folder=paths['toolpaths']
    meta_path=folder/('batch_5axis.meta.json' if (folder/'batch_5axis.meta.json').is_file() else 'batch_3axis.meta.json')
    metadata=json.loads(meta_path.read_text(encoding='utf8'))
    started=time.perf_counter();old=ordered_layers(metadata)
    print(f'[toolpath order] Scheduling {len(old)} saved patches',flush=True)
    layers,report=plan_layers(folder,metadata,config.get('toolpath_ordering',{}).get('enabled',True))
    report['planning_seconds']=time.perf_counter()-started
    # Check the actual newly ordered support starts. If order introduces any
    # finding, retain the existing sequence; this stage never fabricates repair.
    report['support_backing_check_enabled']=bool(config.get('support_paths',{}).get('backing_check',{}).get('enabled',False))
    if report['support_backing_check_enabled'] and meta_path.name=='batch_3axis.meta.json' and metadata.get('support_model',{}).get('same_material') and report['patches_moved']:
        print('[toolpath order] Checking candidate supported starts',flush=True)
        from path_backing_audit import load_records,screen_records
        width=float(metadata.get('width',.5));nominal=float(metadata.get('support_validation',{}).get('nominal_layer_height',.2))
        previous=metadata.get('source_path_metadata',{}).get('support_backing',{}).get('after')
        before=previous if isinstance(previous,dict) and 'findings' in previous else screen_records(load_records(folder,old,workers=4),width=width,nominal=nominal,threads=4)
        after=screen_records(load_records(folder,layers,workers=4),width=width,nominal=nominal,threads=4)
        report['backing_before']=before['status'];report['backing_after']=after['status']
        def finding_keys(findings):
            return {(f.get('main'),f.get('sub'),f.get('lo'),f.get('hi'),f.get('kind')) for f in findings}
        if finding_keys(after['findings'])-finding_keys(before['findings']):
            layers=[dict(row,toolpath_order=i) for i,row in enumerate(old)]
            report.update(rejected_reason='Candidate order increased backing findings; original sequence retained.',
                          patches_moved=0,reordered_groups=0,sequence=[list(identity(r)) for r in layers])
        _write(folder/'toolpath_order_backing.json',after)
    ranks={identity(r):r['toolpath_order'] for r in layers}
    metadata.update(layers=layers,toolpath_ordering=report)
    raw_path=raw_folder/'batch_paths.meta.json';raw=json.loads(raw_path.read_text(encoding='utf8'))
    if {identity(r) for r in raw['layers']}!=set(ranks):
        raise ValueError('Raw and height layer identities differ; cannot save one execution schedule.')
    raw['layers']=[dict(r,toolpath_order=ranks[identity(r)]) for r in raw['layers']]
    raw['layers']=ordered_layers(raw);raw['toolpath_ordering']=report
    metadata['source_path_metadata']=raw
    report['seconds']=time.perf_counter()-started
    _write(raw_path,raw);_write(meta_path,metadata)
    for directory in (raw_folder,folder):
        _write(directory/'toolpath_order.json',dict(schema=SCHEMA,layers=[dict(layer_main=r['layer_main'],
            layer_sub=r['layer_sub'],toolpath_order=r['toolpath_order']) for r in layers]))
    print('[toolpath order] '+json.dumps({k:v for k,v in report.items() if k!='sequence'}),flush=True)
    return report


def order_keys(keys, metadata):
    rows=metadata.get('layers')
    if not isinstance(rows,list) or not any(isinstance(r,dict) and 'toolpath_order' in r for r in rows):return list(keys)
    ranks={identity(r):i for i,r in enumerate(ordered_layers(metadata))}
    return sorted(keys,key=lambda k:ranks.get((k.main,k.sub),len(ranks)))


def write_bare_order(folder,metadata,keys):
    rows=metadata.get('layers')
    if not isinstance(rows,list) or not any(isinstance(r,dict) and 'toolpath_order' in r for r in rows):return
    selected={(k.main,k.sub) for k in keys}
    rows=[r for r in ordered_layers(metadata) if identity(r) in selected]
    if selected!={identity(r) for r in rows}:return # legacy lazy metadata keeps legacy behavior
    _write(Path(folder)/'toolpath_order.json',dict(schema=SCHEMA,layers=[dict(layer_main=r['layer_main'],
        layer_sub=r['layer_sub'],toolpath_order=i) for i,r in enumerate(rows)]))
