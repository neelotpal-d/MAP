"""Shared physical-height eligibility and saved-tree pruning."""
import numpy as np


def resolve_minimum_height(value,points,direction,plate):
    if value is None:
        direction=np.asarray(direction,dtype=float)
        direction=direction/np.linalg.norm(direction)
        return max(0.,float(np.min(np.asarray(points)@direction))-float(plate))
    height_mask(np.empty((0,3)),direction,plate,value)
    return float(value)


def height_mask(points, direction, plate, minimum_height=0.):
    if not np.isfinite(minimum_height) or minimum_height < 0:
        raise ValueError("Minimum support height must be finite and nonnegative.")
    direction=np.asarray(direction,dtype=float)
    direction=direction/np.linalg.norm(direction)
    return np.asarray(points) @ direction - float(plate) >= minimum_height-1e-9


def retained_routes(points, lines, keep):
    """Follow retained contacts through shared trunks, pruning unused branches.

    Saved route polylines stop at earlier trunks and may duplicate point ids at
    merges, so connect by coordinates rather than by ids alone.
    """
    keys=[tuple(p) for p in np.round(np.asarray(points,dtype=float),8)]
    canonical={key:i for i,key in enumerate(keys)}
    parent={}
    for line in lines:
        for a,b in zip(line[:-1],line[1:]):
            a,b=keys[a],keys[b]
            if a==b:
                continue
            if a in parent and parent[a]!=b:
                raise ValueError("Saved tree has conflicting downstream connections.")
            parent[a]=b
    result=[]
    for i in np.flatnonzero(keep):
        key=keys[lines[i][0]]
        path=[]
        seen=set()
        while True:
            if key in seen:
                raise ValueError("Cycle in saved tree.")
            seen.add(key)
            path.append(canonical[key])
            if key not in parent:
                break
            key=parent[key]
        result.append(np.asarray(path,dtype=int))
    return result
