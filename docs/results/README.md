# Current validation evidence

Only current reports are retained here. Timings and image metrics use the exact
DLL checksums in `comparison.json`. The result means the run completed with
finite outputs; it does not establish perceptual quality.

| File | Evidence |
|---|---|
| `comparison.json` | Native timings, per-frame differences, masks, temporal residuals, versions and DLL/source hashes |
| `backend-audit.json` | Actual executed graphs, pinned Python code and official native binary provenance |
| `unified-contracts.json` | Both final builds, regression results and binary hashes |
| `fast-contracts.json`, `contract-tests.json` | Fast and exact contract results |
| `fastest-contracts.json` | Fastest parity, options, budgets, concurrency, fallback and current binary hashes |
| `fastest-dark-line.json` | Faint synthetic line beside a bright patch; exact .03 and fastest .025 thresholds |
| `fastest-video-parity.json` | Final fastest CPU/CUDA pixel parity and exact similarity on 29 video frames |
| `validation.json` | Nonuniform synthetic original-reference stage checks |
| `known-differences.json` | Current numerical compatibility limits |
| `comparison-images.json` | Reviewed PNG hashes, current slow.pics image URLs and verified download hashes |

Source videos and local dependency paths are excluded. Downloaded tools,
unpublished release archives and private upload credentials remain ignored.
