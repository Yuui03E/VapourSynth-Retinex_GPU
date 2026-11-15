"""Thin wrappers around the unified native API 4 plugin."""
from __future__ import annotations
from collections.abc import Sequence
import vapoursynth as vs

def retinex_edgemask_gpu(
    clip: vs.VideoNode,
    sigma: Sequence[float] = (50., 200., 350.),
    lower_thr: float = .001,
    upper_thr: float = .005,
    blur_sigma: float = 1.,
    fast_pyramid: bool = False,
    *,
    op: int = 1,
    device: int = 0,
    workers: int = 2,
    backend: str = "auto",
) -> vs.VideoNode:
    """Return a GRAY mask at the source depth; YUV uses its Y plane only.

    op=1 is kagefunc's Prewitt default; op=2 selects Sobel.
    fast_pyramid=True trades accuracy for speed. Float MSRCP is an extension.
    Load retinex_gpu.dll with core.std.LoadPlugin or install it in plugins first.
    """
    return vs.core.retinex_gpu.RetinexEdgeMask(
        clip, sigma=list(sigma), lower_thr=lower_thr, upper_thr=upper_thr,
        blur_sigma=blur_sigma, fast_pyramid=int(fast_pyramid), op=op,
        device=device, workers=workers, backend=backend, mode="exact",
    )


def retinex_fast(clip: vs.VideoNode, *, backend: str = 'auto', **kwargs) -> vs.VideoNode:
    """Fast MSRCP with the unified DLL."""
    return vs.core.retinex_gpu.MSRCP(clip, mode='fast', backend=backend, **kwargs)

def retinex_edgemask_fast(clip: vs.VideoNode, *, backend: str = 'auto', **kwargs) -> vs.VideoNode:
    """Fast complete mask with the unified DLL."""
    return vs.core.retinex_gpu.RetinexEdgeMask(clip, mode='fast', backend=backend, **kwargs)


def retinex_fastest(clip: vs.VideoNode, *, backend: str = 'auto', **kwargs) -> vs.VideoNode:
    """Fastest local-illumination approximation; fixed downscale 8."""
    return vs.core.retinex_gpu.MSRCP(clip, mode='fastest', backend=backend, **kwargs)


def retinex_edgemask_fastest(clip: vs.VideoNode, *, backend: str = 'auto', **kwargs) -> vs.VideoNode:
    """Fastest complete mask; CPU uses concurrent frame workspaces."""
    return vs.core.retinex_gpu.RetinexEdgeMask(clip, mode='fastest', backend=backend, **kwargs)
