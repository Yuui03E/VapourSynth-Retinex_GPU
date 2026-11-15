"""Independent NumPy CPU extension; upstream MSRCP has no float support.

Literal recurrence, histogram and range rules from the audited CPU source.
This is a correctness oracle for small frames, not a benchmark competitor.
"""
import math
import numpy as np

def gaussian(src, sigma):
    q = 3.97156-4.14554*math.sqrt(1-.26891*sigma) if sigma<2.5 else .98711*sigma-.96330
    b0=1.57825+2.44413*q+1.4281*q*q+.422205*q*q*q
    b1=2.44413*q+2.85619*q*q+1.26661*q*q*q
    b2=-(1.4281*q*q+1.26661*q*q*q)
    b3=.422205*q*q*q
    b=1-(b1+b2+b3)/b0
    b1,b2,b3=b1/b0,b2/b0,b3/b0
    h,w=src.shape
    out=src.copy()
    a=c=d=src[:,0].copy()
    for x in range(1,w):
        z=b*src[:,x]+b1*a+b2*c+b3*d
        d,c,a=c,a,z
        out[:,x]=z
    a=c=d=out[:,-1].copy()
    for x in range(w-2,-1,-1):
        z=b*out[:,x]+b1*a+b2*c+b3*d
        d,c,a=c,a,z
        out[:,x]=z
    for y in range(h):
        out[y]=b*out[y]+b1*out[max(y-1,0)]+b2*out[max(y-2,0)]+b3*out[max(y-3,0)]
    for y in range(h-1,-1,-1):
        out[y]=b*out[y]+b1*out[min(y+1,h-1)]+b2*out[min(y+2,h-1)]+b3*out[min(y+3,h-1)]
    return out

def msrcp_array(src, sigma=(50.,200.,350.), lower=.001, upper=.005):
    src=np.clip(np.where(np.isfinite(src),src,0).astype(np.float64),0,1)
    if all(s==0 for s in sigma):
        return src.astype(np.float32)
    lo,hi=src.min(),src.max()
    norm=(src-lo)*(1/(hi-lo)) if hi>lo else src.copy()
    product=np.ones_like(norm)
    for s in sigma:
        if s==0:
            product*=2
        else:
            blur=gaussian(norm,s)
            ratio=np.ones_like(norm)
            np.divide(norm,blur,out=ratio,where=blur>0)
            product*=np.where(blur>0,ratio+1,1)
    out=np.log(product)/len(sigma)
    lo,hi=out.min(),out.max()
    if hi<=lo:
        return norm.astype(np.float32)
    if lower>0 or upper>0:
        gain=4095/(hi-lo)
        bins=np.clip((out*gain-lo*gain).astype(np.int32),0,4095)
        hist=np.bincount(bins.ravel(),minlength=4096)
        a=np.searchsorted(np.cumsum(hist),int(out.size*lower+.5),side='right')
        b=4095-np.searchsorted(np.cumsum(hist[::-1]),int(out.size*upper+.5),side='right')
        gain=(hi-lo)/4095
        lo,hi=a*gain+lo,b*gain+lo
    if hi<=lo:
        return norm.astype(np.float32)
    gain=1/(hi-lo)
    return np.clip(out*gain-lo*gain,0,1).astype(np.float32)

def float_msrcp(src, sigma, lower, upper):
    def run(n,f):
        out=f.copy()
        np.copyto(np.asarray(out[0]),msrcp_array(np.asarray(f[0]),sigma,lower,upper))
        return out
    return src.std.ModifyFrame(src,run)
