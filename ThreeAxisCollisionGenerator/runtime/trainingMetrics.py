"""Detached epoch telemetry; no extra field evaluations or backward passes."""
import json
from pathlib import Path
import torch

NAMES=('base_direction','base_scalar','surface_direction','support_free','group_scalar','edge_scalar',
       'gradient_ratio','gradient_floor','build_slope','input_curvature','gradient_variation','small_gradient','collision','stress')


class EpochLossAccumulator:
    def __init__(self):self.sums=None;self.count=0
    def add(self,components):
        # One small detached vector addition, rather than a device synchronization
        # per term/batch. Stress is supplied as its actual weighted contribution.
        values=torch.stack([components[name].detach().reshape(()) for name in NAMES])
        self.sums=values if self.sums is None else self.sums+values
        self.count+=1
    def record(self,epoch,total,weights,learning_rate,phase='main'):
        means=self.sums.div(self.count).cpu().tolist() if self.count else [0.]*len(NAMES)
        raw=dict(zip(NAMES,means));weighted={name:raw[name]*weights[name] for name in NAMES}
        weighted['stress']=raw['stress']
        raw['stress']=raw['stress']/weights['stress'] if weights['stress'] else 0.
        return dict(event='train',schema='map.losses.v1',epoch=epoch,total=total,steps=self.count,phase=phase,
                    raw=raw,weighted=weighted,weights=weights,learning_rate_used=learning_rate)


def effective_weights(module,collision_weight,scalar_factor,strength_provider,frame_only=False):
    weights=dict(base_direction=module.BASE_DIRECTION_WEIGHT if module.ENABLE_BASE_DIRECTION_LOSS else 0.,
        base_scalar=module.BASE_SCALAR_WEIGHT if module.ENABLE_BASE_SCALAR_LOSS else 0.,
        surface_direction=module.SURFACE_DIRECTION_WEIGHT if module.ENABLE_SURFACE_DIRECTION_LOSS else 0.,
        support_free=module.SUPPORT_FREE_SURFACE_WEIGHT if module.ENABLE_SUPPORT_FREE_SURFACE_LOSS else 0.,
        group_scalar=scalar_factor*module.SURFACE_GROUP_SCALAR_WEIGHT if module.ENABLE_SURFACE_GROUP_SCALAR_LOSS and not module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS else 0.,
        edge_scalar=scalar_factor*module.SURFACE_EDGE_SCALAR_WEIGHT if module.ENABLE_SURFACE_EDGE_SCALAR_LOSS or module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS else 0.,
        gradient_ratio=module.GRADIENT_BAND_RATIO_WEIGHT,gradient_floor=module.GRADIENT_BAND_FLOOR_WEIGHT,
        build_slope=module.BUILD_DIRECTION_SLOPE_WEIGHT if module.ENABLE_BUILD_DIRECTION_SLOPE_LOSS else 0.,
        input_curvature=module.INPUT_CURVATURE_WEIGHT if module.ENABLE_INPUT_CURVATURE_LOSS else 0.,
        gradient_variation=getattr(module,'GRADIENT_VARIATION_WEIGHT',0.) if module.ENABLE_INPUT_CURVATURE_LOSS else 0.,
        small_gradient=getattr(module,'SMALL_GRADIENT_WEIGHT',0.) if module.ENABLE_INPUT_CURVATURE_LOSS else 0.,
        collision=collision_weight if module.ENABLE_COLLISION_LOSS else 0.,
        stress=float(strength_provider.weight) if strength_provider is not None else 0.)
    if frame_only:weights={name:(collision_weight if name=='collision' else 0.) for name in NAMES}
    return {name:float(value) for name,value in weights.items()}


def write_epoch(checkpoint,row):
    if checkpoint is None:return
    path=Path(checkpoint).parent/'training_metrics.jsonl'
    try:
        with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
    except OSError as error:print(f'[Loss report warning] Could not save epoch metrics: {error}',flush=True)
