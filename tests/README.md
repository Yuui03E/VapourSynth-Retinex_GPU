# Verification tools

Run from the repository root. [Build instructions](../docs/building.md) explain
the optional upstream CPU adapters and failure-injection test build.

| Tool | Purpose |
|---|---|
| `test_unified.py` | Both builds, shared API, exact CPU/CUDA references, constant-input and tiny-blur regressions, fallback |
| `test_fastest.py` | Fastest formats/parity, concurrent frames, fallback, sampled-percentile options and synthetic dark lines |
| `test_fast.py` | Fast CPU/CUDA arithmetic, formats, budgets, scheduling and simulated failures |
| `test_contract.py` | Exact parameters, sanitization, percentile exceptions, wrapper and concurrency |
| `verify.py` | Four exact stages against original CPU code across five depths; excludes the unstable uniform oracle pattern |
| `audit_backends.py` | Execute all 28 native candidate/stage graphs; verify pinned Python sources and official native DLLs |
| `compare_implementations.py` | Native Atom and VSJet CPU/CUDA/OpenCL versus current DLLs; timings, differences and PNGs |
| `benchmark_fastest.py` | Optional native worker-count scaling sweep and dark mask agreement |
| `compare_quality.py` | Shared normalization and metrics used by the comparison harness |
| `float_reference.py` | Independent float extension of the original integer-only Retinex algorithm |

Reports go to ignored `build/reports/`. Curated results are under `docs/results/`.
There is no perceptual ground truth. Simulated failures do not prove recovery
from every physical driver/device loss.

[Reproduce the comparisons](../docs/wiki/Reproduce-the-tests.md).
