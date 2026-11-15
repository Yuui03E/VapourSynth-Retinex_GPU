# Attribution

This project is GPL-3.0-or-later. See LICENSE.

The Retinex equations and recurrence boundary behavior are adapted from
VapourSynth-Retinex, copyright (C) 2014 mawen1250, GPL-3.0-or-later.
The optional CPU verification adapter compiles the upstream Gaussian.cpp and
MSR.cpp function bodies with only their include directives removed.

TCanny formulas and behavior follow VapourSynth-TCanny, VapourSynth port by
HolyWu; original tcanny copyright (C) 2009 Kevin Stone, GPL-3.0-or-later.
The optional reference build compiles that project's unmodified CPU sources.

The filter arrangement and Kirsch matrices follow kagefunc, copyright (c) 2017
kageru, MIT License. Its license is reproduced in licenses/kagefunc-MIT.txt.

VapourSynth headers are supplied by the user's SDK. The GPU DLL statically links
the CUDA runtime and MSVC runtime; NVIDIA's driver is required at runtime.
NVIDIA CUDA tools and downloaded reference repositories are build dependencies
under .tools/ and refs/, not part of the plugin source distribution.

The unified plugin includes exact CPU/CUDA processing and a fast engine with an independent CPU backend and an optional CUDA
backend, under the same GPL-3.0-or-later license. See
[fast dependency notices](src/fast/THIRD_PARTY_NOTICES.md). Generated release
ZIPs may include VapourSynth headers and their original license notices;
the Git repository itself obtains SDK headers from the user's installation.
