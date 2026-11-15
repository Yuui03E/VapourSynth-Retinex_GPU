# Implementation guide

The public architecture and algorithm notes are in
[Implementation](wiki/Implementation.md). Detailed equations, borders and
upstream revisions are in [the reference audit](reference.md).

Keep the exact and fast engines separate behind the single frontend in `src/plugin.cpp`.
Build both DLL variants from the root CMake project. Exact changes must preserve the
audited recurrence order and documented numerical behavior. Fast changes need
both output-difference measurements and matched throughput measurements.

Tests and benchmarks write to ignored `build/reports/`. Curated publication
evidence belongs in `docs/results/`; record hardware, options, versions, hashes,
frame selections and timing exclusions. Compare native masks as different
pipelines, and use common edge stages when isolating a Retinex approximation.

[Build and verify](building.md) · [Test tools](../tests/README.md) ·
[Latest benchmark method](wiki/Benchmark-results.md).
