# MODNet: Material Optimal Descriptor Network

[![arXiv](https://img.shields.io/badge/arXiv-2004.14766-brightgreen)](https://arxiv.org/abs/2004.14766) [![Build Status](https://img.shields.io/github/workflow/status/ppdebreuck/modnet/Run%20tests?logo=github)](https://github.com/ppdebreuck/modnet/actions?query=branch%3Amaster+) ![Read the Docs](https://img.shields.io/readthedocs/modnet)

## Table of contents
- [Introduction](#introduction)
- [How to install](#install)
- [Usage](#usage)
- [Pretrained models](#pretrained)
- [Stored MODData](#stored-moddata)
- [Documentation](#documentation)
- [Getting started](#getting-started)
  - [MODData](#moddata)
  - [MODNetModel](#modnetmodel)
- [Author](#author)
- [License](#license)




<a name="introduction"></a>
## Introduction
This repository contains the Python (3.8) package implementing the Material Optimal Descriptor Network (MODNet).
It is a supervised machine learning framework for **learning material properties** from
either the **composition** or  **crystal structure**. The framework is well suited for **limited datasets**
and can be used for learning *multiple* properties together by using **joint learning**.

This repository also contains two [pretrained models](#pretrained) that can be used for predicting
the refractive index and vibrational thermodynamics from any crystal structure.

See the MODNet papers and repositories below for more details:

- De Breuck *et al.*, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet." *npj Comput Mater* **7**, 83 (2021). [10.1038/s41524-021-00552-2](https://doi.org/10.1038/s41524-021-00552-2) (preprint: [arXiv:2004.14766](https://arxiv.org/abs/2004.14766)).
- De Breuck *et al.*, "Robust model benchmarking and bias-imbalance in data-driven materials science: a case study on MODNet." *J. Phys.: Condens. Matter* **33** 404002,  (2021), [10.1088/1361-648X/ac1280](https://doi.org/10.1088/1361-648X/ac1280) (preprint: [arXiv:2102.02263](https://arxiv.org/abs/2102.02263)).
- MatBench benchmarking data repository: [ml-evs/modnet-matbench](https://github.com/ml-evs/modnet-matbench).



<p align='center'>
<img src="img/MODNet_schematic.PNG" alt="MODNet schematic" />
</p>
<div align='center'>
<strong>Figure 1. Schematic representation of the MODNet.</strong>
</div>


<a name="install"></a>
## How to install

MODNet can be installed via pip:

```bash
pip install modnet
```

<a name="documentation"></a>
## Documentation
The documentation is available at [ReadTheDocs](https://modnet.readthedocs.io).

<a name="author"></a>
## Author
This software were written by [Pierre-Paul De Breuck](mailto:pierre-paul.debreuck@uclouvain.be) and [Matthew Evans](https://www.github.com/ml-evs).

<a name="License"></a>
## License

MODNet is released under the MIT License.
