"""Experimental unified neural field: portable adaptation of the isolated study.

One function is queried in model and support. Model preservation is soft. This
module deliberately exports no optimized_field.npz/grid-router compatibility.
"""
from pathlib import Path
import sys, time, json
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from .platform_router.context import read, save, sha
from .collision_points import PointBoxScalarBVH
from .collision_tool import bounds, frames, contains, pair_membership, COMPONENTS

# Reuse the shipped SIREN implementation, never an experiment folder or env path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'ThreeAxisCollisionGenerator/runtime'))
from siren_pytorch import SirenNet

DEFAULTS = dict(seed=42, cpu_threads=4, width=64, layers=3, frequency=5.,
    affine_steps=500, initialization_steps=1000, rounds=4, steps_per_round=400,
    batch_size=512, smooth_batch=128, model_samples=8192, validation_samples=8192,
    model_value_weight=1000., model_gradient_weight=20., gradient_fit_weight=1.,
    gradient_upper_weight=.2, smoothness_weight=.05, collision_weight=200.,
    learning_rate=2e-4, round_decay=.6, detach_model_collision=True,
    collision_training=True, device='auto')


def configured(settings):
    result=dict(DEFAULTS);result.update({k:v for k,v in settings.items() if k in result})
    for k in ('seed','cpu_threads','width','layers','affine_steps','initialization_steps','rounds','steps_per_round','batch_size','smooth_batch','model_samples','validation_samples'):
        if isinstance(result[k],bool) or not isinstance(result[k],int) or result[k] < (0 if k=='seed' else 1):
            raise ValueError(k+' must be a positive integer (seed may be zero).')
    for k in ('frequency','model_value_weight','model_gradient_weight','gradient_fit_weight','gradient_upper_weight','smoothness_weight','collision_weight','learning_rate','round_decay'):
        if not np.isfinite(result[k]) or result[k] < 0:raise ValueError(k+' must be finite and nonnegative.')
    if min(result['frequency'],result['model_value_weight'],result['learning_rate'],result['round_decay'])<=0 or result['round_decay']>1:
        raise ValueError('Positive frequency, preservation weight and learning rate; decay in (0,1] required.')
    for k in ('detach_model_collision','collision_training'):
        if not isinstance(result[k],bool):raise ValueError(k+' must be a JSON boolean.')
    if result['device']=='auto':result['device']='cuda' if torch.cuda.is_available() else 'cpu'
    if result['device'] not in ('cpu','cuda'):raise ValueError('Choose auto, cpu or cuda.')
    if result['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable.')
    return result


def xyz_grid(d):
    return np.indices(tuple(d['shape'])).reshape(3,-1).T*float(d['spacing'])+d['origin']


class NeuralField(torch.nn.Module):
    def __init__(self, data, seed=42, width=64, layers=3, frequency=5.):
        super().__init__();torch.manual_seed(seed)
        x=xyz_grid(data);center=(x.min(0)+x.max(0))/2;scale=float(np.ptp(x,axis=0).max()/2)
        selected=data['active'].ravel();ref=float(data['reference_gradient'])
        a=np.column_stack(((x[selected]-center)/scale,np.ones(selected.sum())))
        affine=np.linalg.lstsq(a,data['initial'][selected]/ref/scale,rcond=None)[0]
        self.register_buffer('center',torch.tensor(center,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.register_buffer('affine',torch.tensor(affine,dtype=torch.float32))
        self.net=SirenNet(3,width,1,layers,w0=frequency,w0_initial=frequency)
        torch.nn.init.zeros_(self.net.last_layer.weight);torch.nn.init.zeros_(self.net.last_layer.bias)
        self.ref=ref;self.config=dict(seed=seed,width=width,layers=layers,frequency=frequency)

    def forward(self,x):
        z=(x-self.center)/self.scale
        residual=self.net(z,compute_grads=False)['scalars'].reshape(-1)
        return self.scale*(residual+z@self.affine[:3]+self.affine[3])

    def evaluate(self,points,gradient=False,batch=4096):
        points=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('Nonfinite neural query coordinates.')
        if not len(points):return (np.empty(0),np.empty((0,3))) if gradient else np.empty(0)
        values=[];grads=[]
        for start in range(0,len(points),batch):
            x=torch.tensor(points[start:start+batch],dtype=torch.float32,device=self.center.device,requires_grad=gradient)
            with torch.set_grad_enabled(gradient):
                v=self(x)
                if gradient:g=torch.autograd.grad(v.sum(),x)[0]
            values.append(v.detach().cpu().numpy()*self.ref)
            if gradient:grads.append(g.detach().cpu().numpy()*self.ref)
        return (np.concatenate(values),np.concatenate(grads)) if gradient else np.concatenate(values)

    def evaluate_scalars(self,p):return self.evaluate(p)
    def evaluate_scalars_and_gradients(self,p):return self.evaluate(p,True)


def load(folder, device='cpu'):
    folder=Path(folder).resolve();meta=read(folder/'neural_support.json')
    if meta.get('format')!='unified_neural_support_v1':raise ValueError('Not a neural support prototype job.')
    for name,expected in meta['hashes'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder):raise ValueError('Invalid artifact path in neural support manifest.')
        if sha(path)!=expected:raise ValueError('Saved neural support artifact changed: '+name)
    with np.load(folder/'domain.npz') as d:data={k:d[k].copy() for k in d.files}
    ck=torch.load(folder/'network.pt',map_location='cpu',weights_only=True)
    net=NeuralField(data,**ck['config']).to(device);net.load_state_dict(ck['state'])
    return net.eval()


def volume_samples(native,rng,count):
    tet=native.points[native.cells];volume=abs(np.linalg.det(tet[:,1:]-tet[:,:1]))/6
    ids=rng.choice(len(tet),count,p=volume/volume.sum());bary=rng.dirichlet(np.ones(4),count)
    return (np.einsum('ni,nij->nj',bary,tet[ids]),
            np.sum(bary*native.values[native.cells[ids]],axis=1),native.gradients[ids])


def support_mask(points,data,native,gap):
    if not len(points):return np.zeros(0,bool)
    _,closest=native.volume.find_closest_cell(points,return_closest_point=True)
    distance=np.linalg.norm(points-closest,axis=1)
    idx=np.rint((points-data['origin'])/float(data['spacing'])).astype(int)
    idx=np.clip(idx,0,data['shape']-1)
    return (distance>=max(gap,1e-8))&(points[:,1]>0)&(points[:,1]<=data['height'][idx[:,0],idx[:,2]])


def regularization_samples(d,native,gap):
    shape=tuple(d['shape']);active=d['active'].astype(bool);inside=d['inside'].reshape(shape)
    mask=np.ones(np.array(shape)-1,bool);all_inside=mask.copy();targets=[]
    target=d['target'].reshape(shape+(3,))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                s=(slice(i,shape[0]-1+i),slice(j,shape[1]-1+j),slice(k,shape[2]-1+k))
                mask &= active[s];all_inside &= inside[s];targets.append(target[s])
    mask &= ~all_inside
    points=xyz_grid(d).reshape(shape+(3,))
    centers=(points[:-1,:-1,:-1]+float(d['spacing'])/2)[mask]
    targets=np.mean(targets,axis=0)[mask];valid=support_mask(centers,d,native,gap)
    smooth_mask=binary_erosion(active&~inside,iterations=1)[1:-1,1:-1,1:-1]
    smooth=points[1:-1,1:-1,1:-1][smooth_mask]
    smooth=smooth[support_mask(smooth,d,native,gap)]
    if not valid.any():raise ValueError('No support regularization samples; refine spacing or enlarge support shadow.')
    if not len(smooth):smooth=centers[valid].copy()
    return centers[valid],targets[valid],smooth


def search(net,d,support,tool,mode,model=None):
    model=d['model_points'] if model is None else model
    p=dict(model=model,support=support);v={};poses={};report={};pairs=[]
    for key in p:
        if mode=='field_gradient':
            value,g=net.evaluate_scalars_and_gradients(p[key]);poses[key]=frames(g,tool)
        else:value=net.evaluate_scalars(p[key])
        v[key]=np.asarray(value,float)
    trees={k:PointBoxScalarBVH(p[k],v[k]) for k in p};lo,hi=bounds(tool)
    for a,b in [('model','support'),('support','model'),('support','support'),('model','model')]:
        kwargs=dict(frames=poses[a],contains_local=(lambda q:contains(q,tool)) if all(k in tool for k in COMPONENTS) else None) if mode=='field_gradient' else {}
        best,witness,_=trees[b].query(p[a],v[a],lo,hi,**kwargs)
        ids=np.flatnonzero(witness>=0);gap=(v[a][ids]-best[ids])/net.ref
        report[a+'_to_'+b]=dict(roots=len(p[a]),colliding_roots=len(ids),max_order_gap_mm_equiv=float(gap.max(initial=0)))
        if (a,b)!=('model','model'):
            pairs.append(np.column_stack((ids+(len(model) if a=='support' else 0),witness[ids]+(len(model) if b=='support' else 0))))
    return np.vstack(pairs),report


def metrics(net,points,values,gradients):
    v,g=net.evaluate_scalars_and_gradients(points);error=(v-values)/net.ref
    a=np.linalg.norm(gradients,axis=1);b=np.linalg.norm(g,axis=1);valid=(a>1e-10)&(b>1e-10)
    angles=np.degrees(np.arccos(np.clip(np.sum(g[valid]*gradients[valid],axis=1)/(a[valid]*b[valid]),-1,1)))
    return dict(scalar_rms_mm_equiv=float(np.sqrt(np.mean(error**2))),scalar_max_mm_equiv=float(abs(error).max()),
        normal_angle_p95_deg=float(np.percentile(angles,95)) if len(angles) else None,
        local_spacing_change_p95_percent=float(np.percentile(abs(a[valid]/b[valid]-1)*100,95)) if valid.any() else None,
        degenerate_fraction=float(np.mean(~valid)))


def train(pipeline,folder,settings):
    from .conservative_domain import prepare
    from SupportGenerationVoxel.model_field import TetrahedralVertexField
    cfg=configured(settings);torch.set_num_threads(cfg['cpu_threads']);folder=Path(folder)
    started=time.perf_counter();meta=prepare(pipeline,folder,settings)
    meta['field_representation']='unified_neural_prototype';save(folder/'support_field.json',meta)
    with np.load(folder/'domain.npz') as f:d={k:f[k].copy() for k in f.files}
    native=TetrahedralVertexField(Path(meta['model']),meta['scalar_name']);gap=meta['model_clearance_mm']
    ca,tg,sm=regularization_samples(d,native,gap)
    rng=np.random.default_rng(7613);mp,mv,mg=volume_samples(native,rng,cfg['model_samples'])
    hp,hv,hg=volume_samples(native,rng,cfg['validation_samples'])
    with np.load(folder/'holdout.npz') as f:hold=f['points'].copy()
    hold=hold[support_mask(hold,d,native,gap)]
    sp=d['potential_points'];sp=sp[support_mask(sp,d,native,gap)];d['potential_points']=sp
    if not len(sp):raise ValueError('No valid support collision samples.')
    np.savez_compressed(folder/'neural_samples.npz',model_train=mp,model_holdout=hp,model_values=hv,model_gradients=hg,support_train=sp,support_holdout=hold)
    tool=read(folder/'tool.json');mode=meta['collision_mode'];net=NeuralField(d,cfg['seed'],cfg['width'],cfg['layers'],cfg['frequency']).to(cfg['device'])
    def tensor(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32,device=cfg['device'])
    def derivative(x,graph=True):
        x=x.detach().requires_grad_(True);v=net(x);g=torch.autograd.grad(v.sum(),x,create_graph=graph)[0];return x,v,g
    def clock():
        if cfg['device']=='cuda':torch.cuda.synchronize()
        return time.perf_counter()
    def log(**v):print(json.dumps(v),flush=True)
    def pick(n,k,generator):return torch.randperm(n,device=cfg['device'],generator=generator)[:min(n,k)]
    ref=net.ref;ap=tensor(d['anchors']);av=tensor(d['anchor_values']/ref);ag=tensor(d['anchor_gradients']/ref)
    mp=tensor(mp);mv=tensor(mv/ref);mg=tensor(mg/ref);ca=tensor(ca);tg=tensor(tg/ref);sm=tensor(sm)
    x=tensor(xyz_grid(d)[d['active'].ravel()]);target=tensor(d['initial'][d['active'].ravel()]/ref)
    random=torch.Generator(device=cfg['device']).manual_seed(cfg['seed']+1000)
    init_start=clock();opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for step in range(cfg['affine_steps']):
        ids=pick(len(x),2048,random);loss=((net(x[ids])-target[ids])/net.scale).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:log(event='neural_support_initialize',phase='extension_fit',step=step,steps=cfg['affine_steps'],loss=float(loss.detach()))
    for group in opt.param_groups:group['lr']=2e-4
    for step in range(cfg['initialization_steps']):
        ai=pick(len(ap),cfg['batch_size'],random);mi=pick(len(mp),cfg['batch_size'],random);ci=pick(len(ca),256,random)
        _,v,g=derivative(torch.cat((ap[ai],mp[mi],ca[ci])))
        vv=torch.cat((av[ai],mv[mi]));gg=torch.cat((ag[ai],mg[mi]));n=len(vv)
        loss=100*(v[:n]-vv).square().mean()+2*(g[:n]-gg).square().sum(1).mean()+.1*(g[n:]-tg[ci]).square().sum(1).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:log(event='neural_support_initialize',phase='model_preservation',step=step,steps=cfg['initialization_steps'],loss=float(loss.detach()))
    initialization_seconds=clock()-init_start;initial_error=metrics(net,hp,hv,hg)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);random.manual_seed(cfg['seed']+5000)
    all_points=tensor(np.vstack((d['model_points'],sp)));nm=len(d['model_points']);retained=np.empty((0,2),np.int64)
    weights=dict(model_value=cfg['model_value_weight'],model_gradient=cfg['model_gradient_weight'],gradient_fit=cfg['gradient_fit_weight'],gradient_upper=cfg['gradient_upper_weight'],smoothness=cfg['smoothness_weight'],collision=cfg['collision_weight'] if cfg['collision_training'] else 0.)
    history=[];new=np.empty((0,2),np.int64)
    if cfg['collision_training']:new,initial_collisions=search(net,d,sp,tool,mode)
    else:initial_collisions=None
    for replay in range(cfg['rounds']):
        retained=np.unique(np.vstack((retained,new)),axis=0)
        ids=torch.as_tensor(retained,dtype=torch.long,device=cfg['device']);used,inverse=torch.unique(ids.ravel(),return_inverse=True)
        for group in opt.param_groups:group['lr']=cfg['learning_rate']*cfg['round_decay']**replay
        began=clock()
        for step in range(cfg['steps_per_round']):
            ai=pick(len(ap),cfg['batch_size'],random);mi=pick(len(mp),cfg['batch_size'],random);ci=pick(len(ca),cfg['batch_size'],random);si=pick(len(sm),cfg['smooth_batch'],random)
            _,v,g=derivative(torch.cat((ap[ai],mp[mi],ca[ci])))
            vv=torch.cat((av[ai],mv[mi]));gg=torch.cat((ag[ai],mg[mi]));n=len(vv)
            terms=dict(model_value=(v[:n]-vv).square().mean(),model_gradient=(g[:n]-gg).square().sum(1).mean(),gradient_fit=(g[n:]-tg[ci]).square().sum(1).mean(),gradient_upper=torch.relu(g[n:].norm(dim=1)-2).square().mean())
            if weights['smoothness']:
                sx,_,sg=derivative(sm[si]);h=[torch.autograd.grad(sg[:,a].sum(),sx,create_graph=True,retain_graph=True)[0] for a in range(3)]
                terms['smoothness']=25*torch.stack(h,dim=1).square().sum((1,2)).mean()
            else:terms['smoothness']=v.new_zeros(())
            terms['collision']=v.new_zeros(())
            if cfg['collision_training'] and len(used):
                values=net(all_points[used]);effective=torch.where(used<nm,values.detach(),values) if cfg['detach_model_collision'] else values
                pairs=effective[inverse].reshape(-1,2);penalty=torch.relu(pairs[:,0]-pairs[:,1]+.05).square()
                if mode=='field_gradient':
                    _,_,pose_gradient=derivative(all_points[used],graph=False)
                    index=inverse.reshape(-1,2)
                    penalty=penalty*pair_membership(all_points[used],pose_gradient.detach(),index[:,0],index[:,1],tool)
                terms['collision']=penalty.mean()
            loss=sum(weights[k]*value for k,value in terms.items())
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite neural support objective; no completed checkpoint exported.')
            opt.zero_grad();loss.backward();opt.step()
            if step%100==0:log(event='neural_support_train',round=replay+1,step=step,steps=cfg['steps_per_round'],loss=float(loss.detach()))
        train_seconds=clock()-began
        if cfg['collision_training']:new,report=search(net,d,sp,tool,mode)
        else:report=None
        history.append(dict(round=replay+1,train_seconds=train_seconds,pairs=len(retained),collision=report,terms={k:float(v.detach()) for k,v in terms.items()}))
        save(folder/'neural_history.json',history);log(event='neural_support_round_complete',**history[-1])
    net.eval();_,final_training=search(net,d,sp,tool,mode)
    _,final_validation=search(net,d,np.vstack((sp,hold)),tool,mode,model=np.vstack((d['model_points'],hp)))
    checkpoint=folder/'network.pt';torch.save(dict(state=net.state_dict(),config=net.config),checkpoint)
    report=dict(experimental=True,representation='one continuous neural field in model and support',settings=cfg,weights=weights,
        collision_mode=mode,initialization_seconds=initialization_seconds,total_seconds=clock()-started,
        initial_model_error=initial_error,model_holdout=metrics(net,hp,hv,hg),surface_preservation=metrics(net,d['validation'],d['validation_values'],d['validation_gradients']),
        initial_training_collisions=initial_collisions,training_collisions=final_training,validation_collisions=final_validation,
        collision_counts_definition='Sum of directed colliding roots is not unique locations or physical collisions.',
        model_source_unchanged=sha(meta['model'])==meta['model_sha256'],model_predictions_frozen=False,
        downstream_supported=False,physical_validation=False,parameters=sum(p.numel() for p in net.parameters()),
        sampling=dict(model_training=len(mp),model_holdout=len(hp),anchors=len(ap),support_train=len(sp),support_holdout=len(hold),regularization=len(ca),hessian_pool=len(sm)),
        limitations=['Model preservation is soft; inspect scalar and gradient drift.',
                    'Finite potential-volume collision checks, no machine kinematics or continuous guarantee.',
                    'Model-to-model collisions are monitored; only support-related pairs enter collision loss.',
                    'Gradient-following tool membership is a hard active-set check, not a smooth pose derivative.',
                    'Mesh preview samples the network at vertices; it is not direct neural slicing.',
                    'Prototype cannot feed the production grid router/layer adapter.'])
    if not report['model_source_unchanged']:raise RuntimeError('Source model changed during neural optimization.')
    save(folder/'neural_report.json',report)
    import pyvista as pv
    preview=native.volume.copy(deep=True);preview['reference_scalar']=np.asarray(preview[meta['scalar_name']]).copy()
    pv_values,pv_gradients=net.evaluate_scalars_and_gradients(preview.points)
    preview['neural_scalar']=pv_values;preview['neural_gradient']=pv_gradients;preview.set_active_scalars('neural_scalar')
    preview.field_data['neural_support_preview']=[1]
    preview.save(folder/'model_preview.vtu')
    ids=np.linspace(0,len(sp)-1,min(30000,len(sp)),dtype=int);support=pv.PolyData(sp[ids])
    support.field_data['neural_support_preview']=[1]
    support['neural_scalar'],support['neural_gradient']=net.evaluate_scalars_and_gradients(support.points);support.save(folder/'support_preview.vtp')
    artifacts=('network.pt','domain.npz','neural_samples.npz','tool.json','model_preview.vtu','support_preview.vtp','neural_report.json','neural_history.json')
    save(folder/'neural_support.json',dict(format='unified_neural_support_v1',experimental=True,downstream_supported=False,
        display_axis=settings.get('display_axis','y'),collision_mode=mode,source_model=meta['model'],source_model_sha256=meta['model_sha256'],
        settings=cfg,hashes={name:sha(folder/name) for name in artifacts}))
    log(event='neural_support_complete',folder=str(folder),model_holdout=report['model_holdout'],seconds=report['total_seconds'])
    return report
