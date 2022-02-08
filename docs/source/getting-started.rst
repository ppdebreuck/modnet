Getting started
===============

The MODNet package is built around two classes: `MODData` and `MODNetModel`.

Usage
-----

The usual workflow is as follows:

.. code-block:: python

    from modnet.preprocessing import MODData
    from modnet.models import MODNetModel

    # Creating MODData
    data = MODData(materials = structures,
                   targets = targets,
                  )
    data.featurize()
    data.feature_selection(n=200)

    # Creating MODNetModel
    model = MODNetModel(target_hierarchy,
                        weights,
                        num_neurons=[[256],[64],[64],[32]],
                        )
    model.fit(data)

    # Predicting on unlabeled data
    data_to_predict = MODData(new_structures)
    data_to_predict.featurize()
    df_predictions = model.predict(data_to_predict) # returns dataframe containing the prediction on new_structures


Example Notebooks
-----------------

Example notebooks and short tutorials can be found in the *example_notebooks* directory on the `GitHub repo <https://github.com/ppdebreuck/modnet>`_.

Pretrained Models
-----------------

Two pretrained models are provided in *pretrained/*:
    - Refractive index
    - Vibrational thermodynamics

Download these models locally to *path/to/pretrained/*.
Pretrained models can then be used as follows:

.. code-block:: python

    from modnet.models import MODNetModel

    model = MODNetModel.load('path/to/pretrained/refractive_index')
    # or MODNetModel.load('path/to/pretrained/vib_thermo')



Stored MODData
--------------

The following MODDatas are available for download:
    - Formation energy on Materials Project (June 2018), on `figshare <https://figshare.com/articles/dataset/Materials_Project_MP_2018_6_MODData/12834275>`_
    - Refractive index (upon request)
    - Vibrational thermodynamics (upon request)

Download this directory locally to *path/to/moddata/*. These can then be used as follows:

.. code-block:: python

    from modnet.preprocessing import MODData

    data_MP = MODData.load('path/to/moddata/MP_2018.6')

The MP MODData on `figshare <https://figshare.com/articles/dataset/Materials_Project_MP_2018_6_MODData/12834275>`_
(MP_2018.6) is very useful for predicting a learned property on all structures from the Materials Project:

.. code-block:: python

    predictions_on_MP = model.predict(data_MP)
