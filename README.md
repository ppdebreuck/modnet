<div align="center">
    <img src="img/modnet_logo.svg" alt="modnet-logo"  width=200>
    <br>

# MODNet: Material Optimal Descriptor Network

[![arXiv](https://img.shields.io/badge/arXiv-2004.14766-brightgreen)](https://arxiv.org/abs/2004.14766) [![Build Status](https://img.shields.io/github/actions/workflow/status/ppdebreuck/modnet/ci.yml?logo=github&branch=main)](https://github.com/ppdebreuck/modnet/actions?query=branch%3Amaster+) [![Read the Docs](https://img.shields.io/readthedocs/modnet)](https://modnet.readthedocs.io/en/latest/)

</div>

<a name="introduction"></a>
## Introduction
This repository contains the Python (3.8+) package implementing the Material Optimal Descriptor Network (MODNet).
It is a supervised machine learning framework for **learning material properties** from
either the **composition** or  **crystal structure**. The framework is well suited for **limited datasets**
and can be used for learning *multiple* properties together by using **joint learning**.

MODNet appears on the [MatBench leaderboard](https://matbench.materialsproject.org/). As of 11/11/2021, MODNet provides the best performance of all submitted models on 7 out of 13 tasks.

This repository also contains two [pretrained models](#pretrained) that can be used for predicting
the refractive index and vibrational thermodynamics from any crystal structure.

See the MODNet papers and repositories below for more details:

- De Breuck *et al.*, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet." *npj Comput Mater* **7**, 83 (2021). [10.1038/s41524-021-00552-2](https://doi.org/10.1038/s41524-021-00552-2) (preprint: [arXiv:2004.14766](https://arxiv.org/abs/2004.14766)).
- De Breuck *et al.*, "Robust model benchmarking and bias-imbalance in data-driven materials science: a case study on MODNet." *J. Phys.: Condens. Matter* **33** 404002,  (2021), [10.1088/1361-648X/ac1280](https://doi.org/10.1088/1361-648X/ac1280) (preprint: [arXiv:2102.02263](https://arxiv.org/abs/2102.02263)).
- MatBench benchmarking data repository: [modl-uclouvain/modnet-matbench](https://github.com/modl-uclouvain/modnet-matbench).



<p align='center'>
<img src="img/MODNet_schematic.PNG" alt="MODNet schematic" />
</p>
<div align='center'>
<strong>Figure 1. Schematic representation of the MODNet.</strong>
</div>


<a name="install"></a>
## How to install

First, create a virtual environment (e.g., named modnet) with Python (3.8+) using
your favourite environment manager (the following instructions use
[conda](https://docs.conda.io/)):

```shell
conda create -n modnet python=3.9
```

Activate the environment:

```shell
conda activate modnet
```

Finally, install MODNet from PyPI with `pip`:

```shell
pip install modnet
```

>**Warning**
>We strongly recommend pinning your Python environment when using MODNet
>across multiple machines, or multiple MODNet versions, as changes to the
>dependencies and sub-dependencies can lead to different values for particular
>features.
>
>This can be achieved with `conda export` or `pip freeze`.

For development (or if you wish to use pinned versions of direct dependencies that
MODNet has been tested with), you can clone this git repository and make an
editable install inside your chosen environment with `pip`:

```shell
git clone git@github.com:ppdebreuck/modnet
cd modnet
conda create -n modnet python=3.9
conda activate modnet
pip install -r requirements.txt  # optionally use pinned requirements
pip install -e .
```


<a name="documentation"></a>
## Documentation
The documentation is available at [ReadTheDocs](https://modnet.readthedocs.io).

<a name="changelog"></a>
## Changelog
A brief changelog can be found in the [release summaries on GitHub](https://github.com/ppdebreuck/modnet/releases).

<a name="author"></a>
## Author
This software was written by [Pierre-Paul De Breuck](mailto:pierre-paul.debreuck@uclouvain.be) and [Matthew Evans](https://www.github.com/ml-evs) with contributions from David Waroquiers and  Gregoire Heymans.
For an up-to-date list, see the [Contributors on GitHub](https://github.com/ppdebreuck/modnet/graphs/contributors).

<a name="License"></a>
## License

MODNet is released under the MIT License.
