"""Fetch the exact source revisions used by the audit and CPU verification build."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'kagefunc':('https://github.com/Irrational-Encoding-Wizardry/kagefunc.git','6b84049e295f35d3f1fe3197d460c7bdb99c0014'),
    'retinex':('https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex.git','6bfbdd429159c85075dc4b08e0ac4e706470916b'),
    'tcanny':('https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny.git','14ac2ceeb59afc7089974d0ae233fe8d0ea183c8'),
    'vapoursynth':('https://github.com/vapoursynth/vapoursynth.git','dd11a9da6f8e2bb24ab4bb084d44bf01fb93612a'),
    'vsjet':('https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack.git','de0ba4a1ab2c1aa1bdf8024e62d551402e2a2cf6'),
    'vszip':('https://github.com/dnjulek/vapoursynth-zip.git','f6a4c86ee086ad51897796da6e92e6c154e0539c'),
    'vszipcu':('https://github.com/dnjulek/vapoursynth-zipcu.git','ca15eff265b81b8f851ca74fed09f1ef22ce7990'),
    'vszipcl':('https://github.com/dnjulek/vapoursynth-zipcl.git','6c64f91055eb1925cfe203de7286b69c04cea75f'),
}
for name,(url,revision) in SOURCES.items():
    path=ROOT/'refs'/name
    if path.exists():
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=revision:raise SystemExit(f'{path} is at {actual}; expected {revision}. Existing checkout left untouched.')
    else:
        path.mkdir(parents=True)
        for args in (['init'],['remote','add','origin',url],['fetch','--depth','1','origin',revision],['checkout','--detach','FETCH_HEAD']):
            subprocess.run(['git','-C',str(path),*args],check=True)
    print(f'{name}: {revision}')
