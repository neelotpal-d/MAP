"""Reserve a gradient-aligned approach before connecting a saved route."""
import numpy as np
from .support_region import retained_routes


def gradient_approaches(points,lines,ends,gradients,radii,length,direction,plate,margin=.25):
    """Trim route heads, returning downstream routes, necks and neck radii.

    Inputs are trusted contact gradients, not model normals. Full downstream
    paths are recovered before trimming so a short twig can use its shared
    trunk. Callers supply trusted gradients from the selected field.
    """
    if not np.isfinite(length) or length<=0:
        raise ValueError('Gradient tip length must be finite and positive.')
    points=np.asarray(points,dtype=float)
    ends=np.asarray(ends,dtype=float)
    gradients=np.asarray(gradients,dtype=float)
    norms=np.linalg.norm(gradients,axis=1)
    if gradients.shape!=ends.shape or not np.isfinite(gradients).all() or np.any(norms<=1e-10):
        raise ValueError('Valid trusted gradients are required for every tip.')
    axes=gradients/norms[:,None]
    full=retained_routes(points,lines,np.ones(len(lines),dtype=bool))
    trimmed=[]
    necks=[]
    neck_radii=[]
    for index,(line,end,axis) in enumerate(zip(full,ends,axes)):
        depth=(end-points[line])@axis
        eligible=np.flatnonzero(depth>=np.maximum(length,np.asarray(radii)[line]+margin))
        if not len(eligible):
            raise ValueError(f'Route {index} has no downstream room for a {length:g} mm gradient tip.')
        line=line[eligible[0]:]
        radius=float(radii[line[0]])
        actual_length=max(length,radius+margin)
        neck=end-actual_length*axis
        if neck@direction<plate-1e-9:
            raise ValueError(f'Gradient tip {index} would extend below the build plate.')
        trimmed.append(line)
        necks.append(neck)
        neck_radii.append(radius)
    return trimmed,np.asarray(necks),np.asarray(neck_radii)
