# coding: utf-8
# Distributed under the terms of the MIT License.

"""This module defines some remote datasets that can be downloaded
into the user's installation.

"""

import os
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from typing import Union

from modnet.utils import LOG


class Usage(Enum):
    _MODData = auto()
    cross_nmi = auto()
    feature_db = auto()


Dataset = namedtuple("Dataset", ("url", "description", "filename", "md5", "usage"))
DATASETS = {
    "MP_2018.6": Dataset(
        url="https://ndownloader.figshare.com/files/24364571",
        description=(
            "A MODData that contains all inorganic compounds from the Materials Project (MP) as of June 2018, "
            "decorated with the DeBreuck2020 featurizer preset."
        ),
        filename="MP_2018.6.zip",
        md5="06280c4e539508bbcc5266f07698f8d1",
        usage=Usage["_MODData"],
    ),
    "MP_2018.6_CROSS_NMI": Dataset(
        url="https://ndownloader.figshare.com/files/25584803",
        description=(
            "Pickled dataframe containing the Normalized Mutual Information (NMI) between matminer features "
            "computed on the Materials Project."
        ),
        filename="features_cross",
        md5="b83e0bd43f71ec53c4d69ee0764acfbe",
        usage=Usage["cross_nmi"],
    ),
    "MP_210321": Dataset(
        url="https://figshare.com/ndownloader/files/33177986",
        description=(
            "A pickled dataframe containing featurized (DeBreuck2020) materials from the Materials Project as of 21 March 2021."
        ),
        filename="feature_database_v2",
        md5="8efaac9b64f11073577396b9fb353f89",
        usage=Usage["feature_db"],
    ),
}


def load_ext_dataset(dataset_name: str, expected_type: Union[Usage, str]):
    """Load one of the preset datasets from the `DATASETS` constant. Will not
    overwrite any existing local data with remote datasets. Checks hashes against
    what is expected and will not depickle if unrecognised.

    Parameters:
        dataset_name: The name (key) of the dataset in `DATASETS`.
        expected_type: A string representing the expected usage of the dataset,
            e.g. `'_MODData'` or `'cross_nmi'` or `'feature_db'`.

    Returns:
        The path to the downloaded or previously installed model.

    """
    import urllib.request
    import urllib.error

    if dataset_name not in DATASETS:
        raise ValueError(
            f"No dataset {dataset_name} found, must be one of {list(DATASETS.keys())}"
        )

    dataset = DATASETS[dataset_name]
    if isinstance(expected_type, str):
        if expected_type == "MODData":
            expected_type = "_MODData"
        expected_type = Usage[expected_type]
    if dataset.usage != expected_type:
        raise ValueError(
            f"Cannot load {dataset_name} as it has the wrong type {dataset.usage}."
        )

    data_dir = Path(__file__).parent.joinpath("data")
    model_path = data_dir.joinpath(dataset.filename)
    if not model_path.is_file():
        LOG.info(
            f"Downloading featurized dataset {dataset_name} from {dataset.url} into {model_path} \n This may take some time..."
        )
        if not data_dir.is_dir():
            os.makedirs(data_dir)

        try:
            zip_file, response = urllib.request.urlretrieve(dataset.url, model_path)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise ValueError(
                f"There was a problem downloading {dataset.url}: {exc.reason}"
            )

    if dataset.md5 is not None:
        from modnet.utils import get_hash_of_file

        file_md5 = get_hash_of_file(model_path, algo="md5")
        if file_md5 != dataset.md5:
            raise RuntimeError(
                f"Precomputed {str(dataset.usage.name.strip('_'))} did not match expected MD5 from {dataset.url}, will not depickle."
                f"\nExpected: {str(dataset.md5)}"
                f"\nReceived: {str(file_md5)}"
            )

    return model_path
