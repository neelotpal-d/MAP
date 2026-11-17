"""Conservative exterior-field optimizer with explicit fixed/gradient tool poses."""

from pathlib import Path

import json

import time

import numpy as np

from .field_grid import basis,GridField

from .collision_points import PointBoxScalarBVH

from .platform_router.context import read,save,sha

from .collision_tool import mode_from_metadata, bounds, frames, contains, pair_membership, COMPONENTS



def progress(stage, **data):

    print(json.dumps(dict(stage=stage, **data)), flush=True)



def detect(folder,data,values,label,include_centers=False):

    field=GridField(data,values); points=data['potential_points']

    if include_centers:

        points=np.vstack((points,np.load(folder/'holdout.npz')['points']))

    sv,sg=field.evaluate_scalars_and_gradients(points)

    model=data['model_points']; mv=data['model_values']; meta=read(folder/'support_field.json');tool=read(meta['tool'])

    mode=mode_from_metadata(meta);rotating=mode=='field_gradient';lo,hi=bounds(tool)

    shaped=all(k in tool for k in COMPONENTS)
    poses=dict(model=frames(data['model_gradients'],tool),support=frames(sg,tool)) if rotating else {}
    if shaped and not rotating:
        poses=dict(model=np.broadcast_to(np.eye(3),(len(model),3,3)),support=np.broadcast_to(np.eye(3),(len(points),3,3)))

    trees=dict(model=PointBoxScalarBVH(model,mv),support=PointBoxScalarBVH(points,sv))

    all_points=dict(model=model,support=points); all_values=dict(model=mv,support=sv)

    report={}; pairs=[]; nm=len(model); started=time.perf_counter()

    directions=[('model','support'),('support','model'),('support','support')]

    if rotating:directions.append(('model','model'))  # immutable source: report, do not optimize

    for root_kind,target_kind in directions:

        name=f'{root_kind}_to_{target_kind}'

        def announce(done,total,seconds):

            if done==total or seconds>announce.last+15:

                progress('collision_search',label=label,pair=name,done=done,total=total,seconds=seconds); announce.last=seconds

        announce.last=0.

        options=dict(frames=poses[root_kind],contains_local=(lambda x:contains(x,tool)) if all(k in tool for k in COMPONENTS) else None) if (rotating or shaped) else {}

        best,witness,timing=trees[target_kind].query(all_points[root_kind],all_values[root_kind],lo,hi,progress=announce,**options)

        ids=np.flatnonzero(witness>=0); target_ids=witness[ids]

        gap=(all_values[root_kind][ids]-best[ids])/float(data['reference_gradient'])

        relative=all_points[target_kind][target_ids]-all_points[root_kind][ids]

        if rotating:relative=np.einsum('ni,nij->nj',relative,poses[root_kind][ids])

        depths=np.minimum(relative-lo,hi-relative).min(axis=1)

        report[name]=dict(roots=len(all_points[root_kind]),colliding_roots=len(ids),

            scalar_order_gap_reference_mm=dict(maximum=float(gap.max(initial=0)),rms=float(np.sqrt(np.mean(gap**2))) if len(ids) else 0.),

            minimum_scalar_witness_depth_mm=dict(maximum=float(depths.max(initial=0)),

                count_over_point3=int(np.count_nonzero(depths>.3))),**timing)

        rr=ids+(nm if root_kind=='support' else 0); tt=target_ids+(nm if target_kind=='support' else 0)

        if root_kind!=target_kind or root_kind!='model':pairs.append(np.column_stack((rr,tt)))

        if all(k in tool for k in COMPONENTS):

            report[name].pop('minimum_scalar_witness_depth_mm')  # box depth is not primitive penetration

        report[name]['collision_mode']=mode

        # Check a reproducible subset against brute force; membership and scalar ordering both matter.

        check=np.linspace(0,len(all_points[root_kind])-1,12,dtype=int)

        for i in check:

            p=all_points[target_kind]; inside=np.all(p>=all_points[root_kind][i]+lo,axis=1)&np.all(p<=all_points[root_kind][i]+hi,axis=1)

            if rotating or shaped:inside=contains((p-all_points[root_kind][i])@poses[root_kind][i],tool)

            expected=min(all_values[root_kind][i]-1e-8,all_values[target_kind][inside].min(initial=np.inf))

            assert abs(expected-best[i])<1e-10

    save(folder/f'{label}_collisions.json',dict(pairs=report,total_seconds=time.perf_counter()-started,

        potential_samples=len(points),model_samples=len(model),brute_force_spot_checks=12*len(directions),collision_mode=mode))

    np.savez_compressed(folder/f'{label}_witnesses.npz',pairs=np.vstack(pairs),support_values=sv,support_gradients=sg)

    return np.vstack(pairs),report



def optimize(folder,rounds=4,iterations=200,collision_weight=200.,settings=None):

    settings=dict(settings or {})
    collision_training=settings.get('collision_training',True)
    if not isinstance(collision_training,bool):raise ValueError('collision_training must be a JSON boolean.')

    import torch

    from scipy.ndimage import binary_erosion

    if (folder/'optimization_complete.json').exists():

        raise FileExistsError('Completed optimization exists; choose a fresh output folder.')

    started=time.perf_counter(); data=np.load(folder/'domain.npz'); shape=tuple(data['shape']); ref=float(data['reference_gradient'])

    meta=read(folder/'support_field.json');mode=mode_from_metadata(meta);tool=read(meta['tool'])

    if settings.get('collision_mode',mode)!=mode:raise ValueError('Collision mode differs from prepared domain; prepare a fresh job.')

    device=settings.get('device','auto')

    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'

    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable.')

    dtype=torch.float64

    def tensor(a,kind=dtype): return torch.as_tensor(np.asarray(a),dtype=kind,device=device)

    def make_basis(points):

        i,w,d=basis(points,data['origin'],float(data['spacing']),shape)

        return tensor(i,torch.long),tensor(w),tensor(d)

    fixed=tensor(data['initial']); free=tensor(data['free'].astype(float)); h=float(data['spacing'])

    # Optimize corrections in mm-equivalent scalar units, preserving original model nodes exactly.

    variable=torch.nn.Parameter(torch.zeros_like(fixed))

    warm_path=folder/'warm_start.npz'

    if warm_path.exists():

        warm=np.load(warm_path)['values']

        assert np.array_equal(warm[data['inside']],data['initial'][data['inside']])

        with torch.no_grad(): variable.copy_(tensor((warm-data['initial'])/ref)*free)

    target=tensor(data['target'].reshape(shape+(3,)))/ref

    active=data['active'].astype(bool); inside=data['inside'].reshape(shape)

    cellmask=active[:-1,:-1,:-1].copy(); cell_inside=np.ones_like(cellmask)

    for i in (0,1):

        for j in (0,1):

            for k in (0,1):

                s=(slice(i,shape[0]-1+i),slice(j,shape[1]-1+j),slice(k,shape[2]-1+k))

                cellmask &= active[s]; cell_inside &= inside[s]

    cellmask &= ~cell_inside

    cellmask_t=tensor(cellmask,torch.bool)

    # Hessian smoothing is a scalar-field proxy for curvature, not normalized geometric curvature.

    curvature_mask=binary_erosion(active & ~inside,iterations=1)[1:-1,1:-1,1:-1]

    curvature_t=tensor(curvature_mask,torch.bool)

    target_cell=sum(target[i:shape[0]-1+i,j:shape[1]-1+j,k:shape[2]-1+k]

        for i in (0,1) for j in (0,1) for k in (0,1))/8

    ai,aw,ad=make_basis(data['anchors']); av=tensor(data['anchor_values']); ag=tensor(data['anchor_gradients'])/ref

    all_points=np.vstack((data['model_points'],data['potential_points'])); nm=len(data['model_points'])

    pi,pw,pd=make_basis(data['potential_points']); model_values=tensor(data['model_values'])

    collision_points=tensor(all_points) if mode=='field_gradient' and collision_training else None

    model_gradients=tensor(data['model_gradients']) if mode=='field_gradient' and collision_training else None

    active_pairs=np.empty((0,2),dtype=np.int64); pair_r=pair_q=None

    if collision_training and warm_path.exists() and (folder/'active_collision_pairs.npz').exists():

        active_pairs=np.load(folder/'active_collision_pairs.npz')['pairs']

        pair_r=tensor(active_pairs[:,0],torch.long); pair_q=tensor(active_pairs[:,1],torch.long)

    weights=dict(gradient_fit=1.,hessian_smoothing=.05,gradient_upper_bound=.2,

        interface_value=100.,interface_gradient=2.,collision=collision_weight)

    weights.update({k:float(settings[k]) for k in weights if k in settings})

    configured_collision_weight=weights['collision']
    if not collision_training:weights['collision']=0.
    correction_prior=bool(data['correction_prior']) if 'correction_prior' in data else False

    initial_cell_gradient=None

    if correction_prior:

        z=fixed.reshape(shape)/ref

        initial_cell_gradient=torch.stack((

            sum(z[1:,j:shape[1]-1+j,k:shape[2]-1+k]-z[:-1,j:shape[1]-1+j,k:shape[2]-1+k] for j in (0,1) for k in (0,1))/(4*h),

            sum(z[i:shape[0]-1+i,1:,k:shape[2]-1+k]-z[i:shape[0]-1+i,:-1,k:shape[2]-1+k] for i in (0,1) for k in (0,1))/(4*h),

            sum(z[i:shape[0]-1+i,j:shape[1]-1+j,1:]-z[i:shape[0]-1+i,j:shape[1]-1+j,:-1] for i in (0,1) for j in (0,1))/(4*h)

        ),dim=-1)[cellmask_t]

        av=(fixed[ai]*aw).sum(dim=1)

        ag=torch.einsum('ni,nia->na',fixed[ai],ad)/ref

        weights['upward_guard']=2.

    route_file=folder/'route_constraints.npz'; routes=None

    if route_file.exists():

        routes=np.load(route_file); ri,_,rd=make_basis(routes['points'])

        route_up=tensor(routes['up']); route_cos=float(np.cos(np.deg2rad(float(routes['angle_degrees']))))

        normalized_route=bool(routes['normalized_angle']) if 'normalized_angle' in routes else False

        weights.update(route_cone=float(routes['weight']),route_nonzero=float(routes['weight'])/10)

        if 'stage_points' in routes:

            si,_,sd=make_basis(routes['stage_points'])

            stage_contact=tensor(routes['stage_contact_down']); stage_step=tensor(routes['stage_step_down'])

            stage_weight=tensor(routes['stage_weight']); stage_allowed=stage_weight*np.deg2rad(float(routes['angle_degrees']))

            weights['route_stage']=float(routes['weight'])

    material_guard=bool(data['material_gradient_guard']) if 'material_gradient_guard' in data else False

    if material_guard: weights['material_upward']=5000.

    contact_file=folder/'contact_constraints.npz'; contacts=None

    if contact_file.exists():

        contacts=np.load(contact_file); ci,cw,_=make_basis(contacts['points'])

        contact_values=tensor(contacts['model_values']); contact_sign=tensor(contacts['sign'])

        weights['contact_order']=20000.

    history=[]; closures=0; last=time.perf_counter()



    def objective():

        nonlocal closures,last

        phi=fixed+ref*free*variable; f=phi.reshape(shape)/ref

        gx=sum(f[1:,j:shape[1]-1+j,k:shape[2]-1+k]-f[:-1,j:shape[1]-1+j,k:shape[2]-1+k] for j in (0,1) for k in (0,1))/(4*h)

        gy=sum(f[i:shape[0]-1+i,1:,k:shape[2]-1+k]-f[i:shape[0]-1+i,:-1,k:shape[2]-1+k] for i in (0,1) for k in (0,1))/(4*h)

        gz=sum(f[i:shape[0]-1+i,j:shape[1]-1+j,1:]-f[i:shape[0]-1+i,j:shape[1]-1+j,:-1] for i in (0,1) for j in (0,1))/(4*h)

        g=torch.stack((gx,gy,gz),dim=-1)[cellmask_t]

        gradient_fit=((g-(initial_cell_gradient if correction_prior else target_cell[cellmask_t]))**2).sum(dim=1).mean()

        upper=torch.relu(torch.linalg.vector_norm(g,dim=1)-2.) .square().mean()

        smooth=(phi-fixed).reshape(shape)/ref if correction_prior else f

        c=smooth[1:-1,1:-1,1:-1]

        hxx=(smooth[2:,1:-1,1:-1]-2*c+smooth[:-2,1:-1,1:-1])/h**2

        hyy=(smooth[1:-1,2:,1:-1]-2*c+smooth[1:-1,:-2,1:-1])/h**2

        hzz=(smooth[1:-1,1:-1,2:]-2*c+smooth[1:-1,1:-1,:-2])/h**2

        hxy=(smooth[2:,2:,1:-1]-smooth[2:,:-2,1:-1]-smooth[:-2,2:,1:-1]+smooth[:-2,:-2,1:-1])/(4*h**2)

        hxz=(smooth[2:,1:-1,2:]-smooth[2:,1:-1,:-2]-smooth[:-2,1:-1,2:]+smooth[:-2,1:-1,:-2])/(4*h**2)

        hyz=(smooth[1:-1,2:,2:]-smooth[1:-1,2:,:-2]-smooth[1:-1,:-2,2:]+smooth[1:-1,:-2,:-2])/(4*h**2)

        curvature=25*(hxx.square()+hyy.square()+hzz.square()+2*(hxy.square()+hxz.square()+hyz.square()))[curvature_t].mean()

        pred=(phi[ai]*aw).sum(dim=1); gp=torch.einsum('ni,nia->na',phi[ai],ad)/ref

        anchor=((pred-av)/ref).square().mean(); anchor_grad=(gp-ag).square().sum(dim=1).mean()

        collision=torch.zeros((),dtype=dtype,device=device)

        if collision_training and pair_r is not None:

            values=torch.cat((model_values,(phi[pi]*pw).sum(dim=1)))

            violation=(values[pair_r]-values[pair_q])/ref+.05

            penalty=torch.relu(violation).square()

            if mode=='field_gradient':

                # Hard geometry is refreshed each closure; scalar ordering carries

                # gradients. Fresh searches between solves discover new poses/pairs.

                with torch.no_grad():

                    support_gradients=torch.einsum('ni,nia->na',phi[pi],pd)

                    mask=pair_membership(collision_points,torch.cat((model_gradients,support_gradients)),pair_r,pair_q,tool)

                penalty=penalty*mask

            collision=penalty.mean()

        terms=dict(gradient_fit=gradient_fit,hessian_smoothing=curvature,gradient_upper_bound=upper,

            interface_value=anchor,interface_gradient=anchor_grad,collision=collision)

        if correction_prior:

            lower_up=torch.minimum(initial_cell_gradient[:,1],torch.full_like(g[:,1],.5))

            terms['upward_guard']=torch.relu(lower_up-g[:,1]).square().mean()

        if material_guard:

            material_g=torch.einsum('ni,nia->na',phi[pi],pd)/ref

            penalty=torch.relu(.1-material_g[:,1]).square()

            terms['material_upward']=torch.topk(penalty,min(2048,len(penalty)),sorted=False).values.mean()

        if contacts is not None:

            contact_phi=(phi[ci]*cw).sum(dim=1)

            penalty=torch.relu(contact_sign*(contact_phi-contact_values)/ref+.02).square()

            terms['contact_order']=torch.topk(penalty,min(128,len(penalty)),sorted=False).values.mean()

        if routes is not None:

            rg=torch.einsum('ni,nia->na',phi[ri],rd)/ref

            projection=(rg*route_up).sum(dim=1)

            if normalized_route:

                cosine=projection/torch.linalg.vector_norm(rg,dim=1).clamp_min(1e-12)

                angle=torch.acos(cosine.clamp(-1+1e-12,1-1e-12))

                penalty=torch.relu(angle-np.deg2rad(float(routes['angle_degrees']))).square()

            else:

                penalty=torch.relu(route_cos*torch.linalg.vector_norm(rg,dim=1)-projection).square()

            terms['route_cone']=torch.topk(penalty,min(2048,len(penalty)),sorted=False).values.mean()

            small=torch.relu(.1-projection).square()

            terms['route_nonzero']=torch.topk(small,min(2048,len(small)),sorted=False).values.mean()

            if 'stage_points' in routes:

                sg=torch.einsum('ni,nia->na',phi[si],sd)/ref

                local=-sg/torch.linalg.vector_norm(sg,dim=1).clamp_min(1e-12)[:,None]

                theta=torch.acos((stage_contact*local).sum(dim=1).clamp(-1+1e-12,1-1e-12))

                axis=(torch.sin((1-stage_weight)*theta)/torch.sin(theta))[:,None]*stage_contact+(torch.sin(stage_weight*theta)/torch.sin(theta))[:,None]*local

                angle=torch.acos((axis*stage_step).sum(dim=1).clamp(-1+1e-12,1-1e-12))

                stage_penalty=torch.relu((angle-stage_allowed)/stage_weight.clamp_min(1e-4)).square()

                terms['route_stage']=torch.topk(stage_penalty,min(2048,len(stage_penalty)),sorted=False).values.mean()

        total=sum(weights[k]*v for k,v in terms.items())

        closures+=1

        if time.perf_counter()-last>15:

            progress('optimization',closures=closures,pairs=len(active_pairs),loss=float(total.detach()),

                terms={k:float(v.detach()) for k,v in terms.items()},seconds=time.perf_counter()-started)

            last=time.perf_counter()

        return total,terms,phi



    def solve(label):

        # Reset history when the active collision objective changes.

        opt=torch.optim.LBFGS([variable],lr=1.,max_iter=iterations,max_eval=iterations*2,

            history_size=12,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-11)

        def closure():

            opt.zero_grad(); loss,_,_=objective(); loss.backward(); return loss

        before=time.perf_counter(); opt.step(closure)

        with torch.no_grad():

            loss,terms,phi=objective(); values=phi.cpu().numpy()

        assert np.isfinite(values).all()

        assert np.array_equal(values[data['inside']],data['initial'][data['inside']])

        np.savez_compressed(folder/f'{label}_field.npz',values=values)

        row=dict(label=label,seconds=time.perf_counter()-before,total_loss=float(loss),

            terms={k:float(v) for k,v in terms.items()},active_pairs=len(active_pairs))

        history.append(row); save(folder/'optimization_history.json',dict(device=device,weights=weights,history=history,collision_training=collision_training,configured_collision_weight=configured_collision_weight))

        progress('solve_complete',**row)

        return values



    # Meaningful differentiation check: a support-root/model-target violation must lower the root.

    root=torch.tensor(2.,device=device,requires_grad=True); target_value=torch.tensor(1.,device=device)

    torch.relu(root-target_value+.05).square().backward(); assert float(root.grad)>0

    values=solve('regularized_baseline')

    pairs,report=detect(folder,data,values,'regularized_baseline')

    # A disabled collision loss performs one regularization/interface solve.
    # The preceding search is final diagnostics only; no pairs enter training.
    for r in range(1,(rounds if collision_training else 0)+1):

        joined=np.unique(np.vstack((active_pairs,pairs)),axis=0)

        if not len(joined): break

        active_pairs=joined; pair_r=tensor(joined[:,0],torch.long); pair_q=tensor(joined[:,1],torch.long)

        np.savez_compressed(folder/'active_collision_pairs.npz',pairs=active_pairs)

        values=solve(f'round_{r:02d}')

        pairs,report=detect(folder,data,values,f'round_{r:02d}')

    np.savez_compressed(folder/'optimized_field.npz',values=values)

    save(folder/'optimization_complete.json',dict(seconds=time.perf_counter()-started,device=device,

        collision_mode=mode,tool_geometry=tool.get('geometry_kind','box'),
        collision_training=collision_training,configured_collision_weight=configured_collision_weight,
        effective_collision_weight=weights['collision'],collision_diagnostics_performed=True,

        orientation_optimization=('refreshed hard membership; scalar-order gradients; replay mining' if mode=='field_gradient' else 'fixed') if collision_training else 'disabled; final diagnostics only',

        rounds_completed=len(history)-1,weights=weights,active_pairs=len(active_pairs),

        remaining_colliding_roots={k:v['colliding_roots'] for k,v in report.items()},

        support_root_derivative_check_passed=True,hard_model_grid_error=0.,

        model_source_sha256_after=sha(read(folder/'support_field.json')['model'])))

