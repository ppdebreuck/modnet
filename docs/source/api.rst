.. _api:

API
===

MODNet is built around two main classes, `MODData` and `MODNetModel` found in preprocessing and models modules.

* A `MODData` instance is used for representing a particular dataset. It contains a list of structures/compositions and the corresponding target properties, and can be used to perform feature selection.

* A `MODNetModel` instance is used for training and predicting of one or more properties or classes.

The complete hierarchical Python API of modnet can be found below.

.. toctree::
   :maxdepth: 4

   modnet
