

# Setup environment

Ideally, set up a new conda environment and install some lbraries:

``` bash
conda create -n wacv_submission_246 python=3.8.15

conda activate wacv_submission_246

pip install ortools sparse scipy numpy tifffile matplotlib opencv-python scikit-image jupyter networkx pandas
```


# Code 

The metric implemnentation can be found under metric.py

A minimal working example that uses the multi-mask similarity measure on M masks is given below:

(Note: same code also available in minimal-working-example.ipynb )

``` python
from metric import MetricsMultiMask
from maskgen import MaskGenerator

mmm = MetricsMultiMask() # optional: MetricsMultiMask(verbose=False)
mg = MaskGenerator()

M = 4    # 4 Masks
nels = 500    # number of elements per mask
    
masks = []
for m in range(M):
    _, tmp_mask = mg.generate_mask(nels, seed=None)
    masks.append(tmp_mask.copy())

score = mmm.do_fullrun(masks)
print(f'\nAgreement score for {M} random masks: {score}')
```

Note that this minimal working example outputs raw (uncorrected) scores. 


# reimplementation of Durkee's generalizations


``` python
import Durkee

from maskgen import MaskGenerator

durkee = Durkee.MyDurkee()
mg = MaskGenerator()

M = 4    # 4 Masks
nels = 500    # number of elements per mask
    
masks = []
for m in range(M):
    _, tmp_mask = mg.generate_mask(nels, seed=None)
    masks.append(tmp_mask.copy())

score = durkee.do_it(masks)
print(f'\nAgreement score for {M} random masks using Durkee\'s generalizations (SI = sorensen index, JI = jaccard index): {score}')
```


# [Optional] Download real data

If you want to recreate the experiments that employ real mask, make sure to download the openly available masks.

## Sonneck et al.

https://zenodo.org/records/8247136

## Durkee et al.

https://github.com/durkeems13/multireader_metrics/tree/main

## Schuiki et al.

https://zenodo.org/records/15511299 . 

