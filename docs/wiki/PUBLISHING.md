# Updating the Wiki

The public Wiki is at https://github.com/Yuui03E/VapourSynth-Retinex_GPU/wiki.
Its source pages and charts are maintained in this directory. Before publishing
updates, run repository checks and ensure release DLL checksums match the
current evidence in `docs/results/`.

Copy the five Wiki pages and `images/` into the Wiki checkout. Convert links
within the Wiki to page names without `.md`; links to other project files
should point to the source repository. Release ZIPs belong in GitHub Releases.
Build directories, downloaded dependencies and private upload records stay local.
