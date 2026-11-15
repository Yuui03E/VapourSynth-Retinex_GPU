# Third-party notices

The small-image Young/van Vliet Gaussian equations and endpoint behavior derive
from mawen1250's VapourSynth-Retinex (GPL-3.0); the surrounding exact-project
implementation and audit are described in `docs/reference.md`. The companion
source is GPL-3.0-or-later; see the root `LICENSE`.

Bundled VapourSynth API headers retain their own notices:

* `VapourSynth4.h`: Copyright (c) 2012–2026 Fredrik Mellbin,
  LGPL-2.1-or-later. Full license: `refs/vapoursynth/COPYING.LESSER`.
* `VSHelper4.h`: Copyright (c) 2012–2020 Fredrik Mellbin, WTFPL version 2:

```
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
```

The GPU binary statically links NVIDIA's CUDA runtime. CUDA toolkit components
are governed by NVIDIA's CUDA Toolkit End User License Agreement; the full
toolkit and compiler are not included in this package.
