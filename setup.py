import re

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("modnet/__init__.py", "r") as f:
    lines = ""
    for item in f.readlines():
        lines += item + "\n"


version = re.search('__version__ = "(.*)"', lines).group(1)

tests_require = ("pytest~=8.0", "pytest-cov~=5.0", "flake8~=7.0")
dev_require = ("pre-commit~=3.7",)

setuptools.setup(
    name="modnet",
    version=version,
    author="Pierre-Paul De Breuck",
    author_email="pierre-paul.debreuck@uclouvain.be",
    description="MODNet, the Material Optimal Descriptor Network for materials properties prediction. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ppdebreuck/modnet",
    project_urls={
        "GitHub": "https://github.com/ppdebreuck/modnet",
        "Documentation": "https://modnet.readthedocs.io",
    },
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas <= 1.5, < 3",
        "tensorflow ~= 2.10, < 2.16",
        "pymatgen >= 2023",
        "matminer ~= 0.9",
        "numpy >= 1.24",
        "scikit-learn ~= 1.3",
    ],
    tests_require=tests_require,
    test_suite="modnet.tests",
    extras_require={
        "bayesian": ["tensorflow-probability==0.18", "tensorflow == 2.11.*"],
        "test": tests_require,
        "dev": dev_require,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
