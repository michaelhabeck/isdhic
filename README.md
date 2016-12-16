# isdhic #

This Python package implements a minimal version of [Inferential Structure Determination (ISD)](http://science.sciencemag.org/content/309/5732/303) and can be used to calculate structures of chromosomes from single-cell Hi-C data.

### Installation ###

* Local installation
```
python setup.py install --prefix=${HOME}
```

* Global installation
```
sudo python setup.py install
```

### Dependencies ###

* numpy: [download](https://pypi.python.org/pypi/numpy)

* scipy: [download](https://pypi.python.org/pypi/scipy)

* csb:   [download](https://pypi.python.org/pypi/csb)

Install with
```
pip install numpy scipy csb
```

Optional (used in only test and application scripts)

* matplotlib: [download](http://matplotlib.org)

Install with
```
pip install matplotlib
```

### Usage ###

Examples for running ISD on single-cell chromosome
[data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48262)
from
[Nagano *et al.* (2013)](http://www.nature.com/nature/journal/v502/n7469/full/nature12593.html)
can be found in the *scripts/* folder.

Execute the scripts *run_hmc.py* and *analyze.py*. 
