"""Explicit per-job context for promoted algorithms; no model-specific paths."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import copy

_CURRENT=ContextVar('platform_router_context',default=None)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


@dataclass
class RoutingContext:
    neck_source:Path
    output_root:Path
    saved:dict
    field_factory:object
    distance_factory:object
    overrides:dict=field(default_factory=dict)


def current():
    result=_CURRENT.get()
    if result is None:raise RuntimeError('Initialize the platform router with an explicit RoutingContext first.')
    return result


def protected_length():
    c=_CURRENT.get()
    if c is None:return 6.
    settings=c.saved['arguments']
    return float(settings.get('contact_gradient_length',3.))+float(settings.get('contact_transition_length',3.))


@contextmanager
def use_context(context):
    token=_CURRENT.set(context)
    try:yield context
    finally:_CURRENT.reset(token)


class _JobPath(os.PathLike):
    def __init__(self,attribute):self.attribute=attribute
    def __fspath__(self):return os.fspath(getattr(current(),self.attribute))
    def __truediv__(self,child):return Path(self)/child
    def __getattr__(self,name):return getattr(Path(self),name)
    def __str__(self):return os.fspath(self)


SOURCE=_JobPath('neck_source')
CASE=_JobPath('output_root')


def setup():
    import numpy as np
    import pyvista as pv
    c=current();saved=copy.deepcopy(c.saved)
    return (saved,saved['arguments'],pv.read(c.neck_source/'support_centerlines.vtp'),
            c.field_factory(),c.distance_factory(),np.array([0.,1.,0.]),{},{})


def parameters(output):
    options=dict(output=Path(output),source=None,geometry_threads=4,fixed_work=True,
        scalar_baseline=False,zone=24.,max_drop=56.,min_bend_radius=5.,footing_budget=.15,
        component_budget=1.,index_spacing=1.5,candidates_per_anchor=12,fine_candidates=8,
        side_fine_candidates=12,proposal_limit=144,include_subbranches=True,
        fast_field=False,certified_guide=False,legacy_fallback=False)
    options.update(current().overrides)
    return SimpleNamespace(**options)
