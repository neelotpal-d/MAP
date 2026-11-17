"""Optional footprint-screened widening of accepted model roots only."""
import time
import numpy as np
from .landing_checks import tangent_ring


def widen_roots(graph,seed_count,query,normals,evaluator,distance,maximum_radius=4.,
                minimum_alignment=.95,gap=.3,base_radius=1.2,body_maximum=2.5,
                direction=(0,1,0),height_tolerance=.1,footprint_mode='conformal'):
    if not np.isfinite(maximum_radius) or maximum_radius<=0 or not 0<minimum_alignment<1:
        raise ValueError('Invalid model-root foot settings.')
    if footprint_mode not in ('conformal','planar'):raise ValueError('Unknown foot footprint mode.')
    started=time.perf_counter();out=graph.copy(deep=True);points=np.asarray(graph.points)
    edges=graph.lines.reshape(-1,3)[:,1:];parent=np.full(len(points),-1,dtype=int)
    parent[edges[:,0]]=edges[:,1]
    accepted=np.asarray(graph.point_data.get('is_root',np.asarray(graph['node_kind'])==2),dtype=bool)
    stalled=np.asarray(graph.point_data.get('is_stalled',np.asarray(graph['node_kind'])==3),dtype=bool)
    roots=np.flatnonzero(accepted&~stalled&(parent<0))
    saved=np.asarray(graph.point_data.get('root_contact_radius',np.zeros(len(points))),dtype=float).copy()
    transition=np.asarray(graph.point_data.get('root_foot_transition_mm',np.zeros(len(points))),dtype=float).copy()
    # A centred sphere is unnecessarily restrictive beside a nearby wall.  These
    # optional primitives describe a single conformal lobe which is swept back
    # into the unchanged root by ``sweep_rows``.
    lobe_centres=np.asarray(graph.point_data.get('root_foot_lobe_center',np.zeros_like(points)),dtype=float).copy()
    lobe_radii=np.asarray(graph.point_data.get('root_foot_lobe_radius',np.zeros(len(points))),dtype=float).copy()
    load=np.zeros(len(points));load[:seed_count]=1
    shortest=np.full(len(points),np.inf);shortest[:seed_count]=0.
    for node in np.argsort(points@np.asarray(direction))[::-1]:
        p=parent[node]
        if p>=0:
            load[p]+=load[node]
            shortest[p]=min(shortest[p],shortest[node]+np.linalg.norm(points[node]-points[p]))
    records=[];probe_count=0
    for node in roots:
        old=float(saved[node]);body=min(body_maximum,base_radius*np.sqrt(max(1.,load[node])))
        record=dict(node=int(node),old_radius_mm=old,new_radius_mm=old,rejection_counts={})
        def reject(reason):record['rejection_counts'][reason]=record['rejection_counts'].get(reason,0)+1
        centre,ids=query(points[node:node+1]);normal=np.asarray(normals[ids[0]],dtype=float)
        normal/=max(np.linalg.norm(normal),1e-15)
        if abs(float(distance(points[node:node+1])[0])-gap)>.1:
            record['reason']='not_at_model_gap';records.append(record);continue
        ring=tangent_ring(normal[None,:])[0]
        rotated=ring+np.roll(ring,-1,axis=0);rotated/=np.linalg.norm(rotated,axis=1)[:,None]
        ring=np.vstack((ring,rotated))
        chosen=None
        # Bounded trials, retaining the old contact if no larger disk passes.
        radius_trials=(np.linspace(max(old,body),maximum_radius,5)[::-1] if footprint_mode=='planar'
                       else np.unique(np.r_[np.linspace(old,maximum_radius,9),
                            maximum_radius*np.array([1.,.875,.75,.625,.5,.375,.3125,.25,.1875,.15625,.125])])[::-1])
        for radius in radius_trials:
            if radius<=old+1e-6:continue
            length=max(3.,2.*(radius-base_radius)/np.tan(np.deg2rad(40.)))
            if shortest[node]<length:reject('short_trunk');continue
            stencil=np.vstack((np.zeros((1,3)),(np.array([1/3,2/3,1.])[:,None,None]*radius*ring[None,:,:]).reshape(-1,3)))
            plane=centre+stencil
            surface,faces=query(plane);n=np.asarray(normals)[faces]
            values,g=evaluator.evaluate_scalars_and_gradients(surface)
            g=np.asarray(g);norm=np.linalg.norm(g,axis=1);unit=g/np.maximum(norm[:,None],1e-15)
            probe_count+=len(plane)
            if not np.isfinite(g).all() or not np.isfinite(values).all() or np.any(norm<1e-8):reject('invalid_field');continue
            if np.any(np.sum(n*unit,axis=1)<minimum_alignment):reject('field_surface_misalignment');continue
            normal_limit=minimum_alignment if footprint_mode=='planar' else np.cos(np.deg2rad(45.))
            if np.any(n@normal<normal_limit):reject('surface_normal_jump');continue
            projection_limit=height_tolerance if footprint_mode=='planar' else max(height_tolerance,radius*.25)
            if np.max(np.linalg.norm(surface-plane,axis=1))>projection_limit:reject('surface_projection_too_far');continue
            # A conformal disk should be close to one native model layer.
            if np.ptp(values)/max(float(np.median(norm)),1e-15)>2*height_tolerance:reject('scalar_variation');continue
            backing=(plane-.05*normal if footprint_mode=='planar' else surface-.05*n)
            if not np.all(np.asarray(distance(backing))<0):reject('missing_backing');continue
            if footprint_mode=='conformal':
                offset=surface+gap*n
                if np.any(np.asarray(distance(offset))<gap-.02):reject('offset_clearance');continue
                # The inner two rings must fit within the widened root sphere.
                # The outer ring screens nearby surface continuity/alignment;
                # the actual sphere footprint is cut by exact model clearance.
                if np.max(np.linalg.norm(offset[:25]-points[node],axis=1))>radius:
                    reject('insufficient_sphere_overlap');continue
            chosen=(float(radius),float(length),float(np.min(np.sum(n*unit,axis=1))));break
        if chosen is not None:
            saved[node],transition[node],alignment=chosen
            record.update(new_radius_mm=float(saved[node]),transition_mm=float(transition[node]),minimum_alignment=alignment,reason='widened')
        # If every centred disk crosses a nearby wall, try one lobe whose centre
        # moves into a valid tangent direction.  This keeps the actual routed
        # root and trunk fixed, so it cannot introduce a new field-cone failure.
        # The lobe is still a complete, locally conformal disk at its own centre;
        # only the *combined* footprint is asymmetric.
        if footprint_mode=='conformal' and saved[node]<maximum_radius-1e-6:
            lobe_choice=None
            directions=np.vstack((ring,tangent_ring(normal[None,:])[0]))
            for radius in radius_trials:
                if radius<=saved[node]+1e-6:continue
                if shortest[node]<3.:
                    reject('asymmetric_short_trunk');continue
                # Move far enough that the lobe's blocked-side rim is no longer
                # forced through the original root, while retaining overlap with
                # the current root sphere.
                shift=radius+.5*saved[node]
                # Batch every direction for this radius.  The original version
                # issued one VTK nearest-surface query and one neural-field call
                # per direction; this is the same screen with vectorized inputs.
                anchors=np.asarray(centre,dtype=float).reshape(1,3)+shift*directions
                lobe_surface,lobe_ids=query(anchors)
                lobe_normal=np.asarray(normals)[np.asarray(lobe_ids,dtype=int)]
                lobe_normal/=np.maximum(np.linalg.norm(lobe_normal,axis=1)[:,None],1e-15)
                stencil=np.vstack((np.zeros((1,3)),(np.array([1/3,2/3,1.])[:,None,None]*radius*ring[None,:,:]).reshape(-1,3)))
                plane=lobe_surface[:,None,:]+stencil[None,:,:]
                sample_count=len(stencil);flat_plane=plane.reshape(-1,3)
                surface,faces=query(flat_plane);n=np.asarray(normals)[faces]
                values,g=evaluator.evaluate_scalars_and_gradients(surface)
                surface=surface.reshape(len(directions),sample_count,3);n=n.reshape(len(directions),sample_count,3)
                values=np.asarray(values).reshape(len(directions),sample_count)
                g=np.asarray(g).reshape(len(directions),sample_count,3)
                norm=np.linalg.norm(g,axis=2);unit=g/np.maximum(norm[:,:,None],1e-15)
                probe_count+=len(flat_plane)
                valid=np.ones(len(directions),dtype=bool)
                def reject_mask(mask,reason):
                    for _ in np.flatnonzero(mask):reject(reason)
                bad=~np.isfinite(g).all(axis=(1,2))|~np.isfinite(values).all(axis=1)|np.any(norm<1e-8,axis=1)
                reject_mask(bad,'asymmetric_invalid_field');valid&=~bad
                bad=valid&np.any(np.sum(n*unit,axis=2)<minimum_alignment,axis=1)
                reject_mask(bad,'asymmetric_field_surface_misalignment');valid&=~bad
                bad=valid&np.any(np.sum(n*lobe_normal[:,None,:],axis=2)<np.cos(np.deg2rad(45.)),axis=1)
                reject_mask(bad,'asymmetric_surface_normal_jump');valid&=~bad
                bad=valid&(np.max(np.linalg.norm(surface-plane,axis=2),axis=1)>max(height_tolerance,radius*.25))
                reject_mask(bad,'asymmetric_surface_projection_too_far');valid&=~bad
                scalar_span=np.ptp(values,axis=1)/np.maximum(np.median(norm,axis=1),1e-15)
                bad=valid&(scalar_span>2*height_tolerance)
                reject_mask(bad,'asymmetric_scalar_variation');valid&=~bad
                backing=surface-.05*n
                bad=valid&~np.all(np.asarray(distance(backing.reshape(-1,3))).reshape(len(directions),sample_count)<0,axis=1)
                reject_mask(bad,'asymmetric_missing_backing');valid&=~bad
                offset=surface+gap*n
                bad=valid&~np.all(np.asarray(distance(offset.reshape(-1,3))).reshape(len(directions),sample_count)>=gap-.02,axis=1)
                reject_mask(bad,'asymmetric_offset_clearance');valid&=~bad
                lobe_points=lobe_surface+gap*lobe_normal
                bad=valid&(np.linalg.norm(lobe_points-points[node],axis=1)>=radius+saved[node]-1e-4)
                reject_mask(bad,'asymmetric_no_overlap');valid&=~bad
                if np.any(valid):
                    selected=int(np.flatnonzero(valid)[0])
                    lobe_choice=(float(radius),lobe_points[selected].copy(),float(np.min(np.sum(n[selected]*unit[selected],axis=1))))
                if lobe_choice is not None:break
            if lobe_choice is not None:
                radius,lobe_point,alignment=lobe_choice
                lobe_centres[node]=lobe_point;lobe_radii[node]=radius
                record.update(lobe_radius_mm=radius,lobe_center_mm=lobe_point.tolist(),minimum_alignment=alignment,
                              reason='widened_plus_asymmetric_lobe' if chosen is not None else 'widened_asymmetric_lobe')
            elif chosen is None:record['reason']='no_larger_backed_conformal_disk_or_asymmetric_lobe'
        elif chosen is None:
            record['reason']='no_larger_backed_conformal_disk_or_short_trunk'
        records.append(record)
    out['root_contact_radius']=saved;out['root_foot_transition_mm']=transition
    out['root_foot_lobe_center']=lobe_centres;out['root_foot_lobe_radius']=lobe_radii
    return out,dict(checked_roots=len(roots),widened_roots=sum(r['reason'] in ('widened','widened_asymmetric_lobe','widened_plus_asymmetric_lobe') for r in records),
        maximum_radius_mm=float(maximum_radius),minimum_alignment=float(minimum_alignment),
        footprint_mode=footprint_mode,
        height_tolerance_mm=height_tolerance,probe_points=probe_count,seconds=time.perf_counter()-started,
        records=records,limitation='Sampled model-tangent footprint screen; not a full first-layer or swept-volume printability certificate.')
