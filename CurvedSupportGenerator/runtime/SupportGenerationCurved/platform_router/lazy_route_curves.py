"""Reusable router definitions promoted from lazy_route_curves.py; see promotion_sources.json."""
import numpy as np

def horner(coefficients,t):
    result=coefficients[-1,:,None]+np.zeros_like(t)
    for c in coefficients[-2::-1]:result=result*t+c[:,None]
    return result

def squared_coefficients(c,b,a):
    return np.array([np.sum(c*c,axis=1),2*np.sum(c*b,axis=1),
                     np.sum(b*b+2*c*a,axis=1),2*np.sum(a*b,axis=1),np.sum(a*a,axis=1)])

class LazyChain:
    def __init__(self,controls,counts,k):
        self.controls=controls;self.counts=counts
        self.ends=np.cumsum(counts-1);self.k=k;self.materialized=None
    def __len__(self):return int(self.ends[-1]+1)
    def samples(self,indices):
        indices=np.asarray(indices);flat=indices.reshape(-1)
        part=np.searchsorted(self.ends,flat,side='left');starts=np.r_[0,self.ends[:-1]]
        t=(flat-starts[part])/(self.counts[part]-1);u=1.-t
        a,c,d,b=self.controls[part].transpose(1,0,2)
        p=u[:,None]**3*a+3*u[:,None]**2*t[:,None]*c+3*u[:,None]*t[:,None]**2*d+t[:,None]**3*b
        return p.reshape((*indices.shape,3))
    def __getitem__(self,key):
        if isinstance(key,tuple):
            first,*rest=key;ids=np.arange(len(self))[first];p=self.samples(ids)
            return p[tuple(rest)] if np.ndim(ids)==0 else p[(slice(None),*rest)]
        return self.samples(np.arange(len(self))[key])
    def __array__(self,dtype=None,copy=None):
        if self.materialized is None:self.materialized=self.samples(np.arange(len(self)))
        return np.array(self.materialized,dtype=dtype,copy=True) if copy else np.asarray(self.materialized,dtype=dtype)

def classify_chains(points,sizes,first,last,min_radius=5.,chunk_size=128):
    points=np.asarray(points,float);sizes=np.asarray(sizes,int);first=np.asarray(first);last=np.asarray(last)
    if not len(points):return []
    n=len(points);tangents=np.zeros_like(points);tangents[:,0]=first;tangents[np.arange(n),sizes-1]=last
    for j in range(1,points.shape[1]-1):
        ids=np.flatnonzero(sizes>j+1);v=points[ids,j+1]-points[ids,j-1]
        tangents[ids,j]=v/np.linalg.norm(v,axis=1)[:,None]
    valid=np.ones(n,bool);controls=[[] for _ in sizes];counts=[[] for _ in sizes];maximum=np.zeros(n)
    for j in range(points.shape[1]-1):
        available=np.flatnonzero(valid&(sizes>j+1))
        for offset in range(0,len(available),chunk_size):
            ids=available[offset:offset+chunk_size];a=points[ids,j];b=points[ids,j+1]
            t0=tangents[ids,j];t1=tangents[ids,j+1]
            chord=np.linalg.norm(b-a,axis=1);drop=a[:,1]-b[:,1]
            h=np.minimum(.42*chord,.96*drop/np.maximum(abs(t0[:,1])+abs(t1[:,1]),1e-6))
            c=a+h[:,None]*t0;d=b-h[:,None]*t1
            count=np.maximum(25,np.ceil(chord/.15).astype(int)+1)
            # v(t)=A*t^2+B*t+C; v cross acceleration is quadratic.
            C=3*(c-a);B=6*(d-2*c+a);A=3*(b-3*d+3*c-a)
            Q0=np.cross(C,B);Q1=2*np.cross(C,A);Q2=np.cross(B,A)
            speed_coeff=squared_coefficients(C,B,A);cross_coeff=squared_coefficients(Q0,Q1,Q2)
            columns=np.arange(int(count.max()))[None,:]
            t=np.minimum(columns/(count[:,None]-1),1.);u=1.-t;mask=columns<count[:,None]
            speed2=horner(speed_coeff,t);cross2=horner(cross_coeff,t)
            k2=np.maximum(cross2,0.)/np.maximum(np.maximum(speed2,0.)**3,1e-30)
            k2=np.where(mask,k2,0.);maximum_indices=np.argmax(k2,axis=1)
            # Re-evaluate the maximum with the original vector expression.
            tm=maximum_indices/(count-1);um=1.-tm
            velocity=3*um[:,None]**2*(c-a)+6*um[:,None]*tm[:,None]*(d-c)+3*tm[:,None]**2*(b-d)
            acceleration=6*um[:,None]*(d-2*c+a)+6*tm[:,None]*(b-2*d+c)
            k=np.linalg.norm(np.cross(velocity,acceleration),axis=1)/np.maximum(np.linalg.norm(velocity,axis=1)**3,1e-15)
            y=u**3*a[:,1,None]+3*u**2*t*c[:,1,None]+3*u*t**2*d[:,1,None]+t**3*b[:,1,None]
            bad=((np.diff(y,axis=1)>=-1e-8)&(columns[:,:-1]<(count-1)[:,None])).any(axis=1)
            ok=(chord>=.1)&(drop>.01)&(k<=1./min_radius)&~bad
            # Poorly conditioned polynomials and threshold ties use the
            # original vector grid, never a more permissive classification.
            uncertain=((speed2.min(axis=1)<1e-7*np.maximum(abs(speed_coeff).max(axis=0),1.))|
                       (abs(k-1./min_radius)<1e-7))
            for row in np.flatnonzero(uncertain):
                tt=np.linspace(0,1,count[row]);uu=1.-tt
                v=3*uu[:,None]**2*(c[row]-a[row])+6*uu[:,None]*tt[:,None]*(d[row]-c[row])+3*tt[:,None]**2*(b[row]-d[row])
                acc=6*uu[:,None]*(d[row]-2*c[row]+a[row])+6*tt[:,None]*(b[row]-2*d[row]+c[row])
                k[row]=np.max(np.linalg.norm(np.cross(v,acc),axis=1)/np.maximum(np.linalg.norm(v,axis=1)**3,1e-15))
                ok[row]=(chord[row]>=.1 and drop[row]>.01 and k[row]<=1./min_radius and not bad[row])
            for row,i in enumerate(ids):
                if not ok[row]:valid[i]=False;continue
                controls[i].append(np.stack((a[row],c[row],d[row],b[row])))
                counts[i].append(count[row]);maximum[i]=max(maximum[i],k[row])
    return [LazyChain(np.asarray(controls[i]),np.asarray(counts[i]),float(maximum[i])) if valid[i] else None for i in range(n)]
