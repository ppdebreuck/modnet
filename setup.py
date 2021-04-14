import setuptools
import re

with open("README.md", "r") as f:
    long_description = f.read()

with open("modnet/__init__.py", "r") as f:
    lines = ""
    for item in f.readlines():
        lines += item + "\n"


version = re.search('__version__ = "(.*)"', lines).group(1)

tests_require = ["pytest>=6.0", "pytest-cov>=2.10", "flake8>=3.8"]

dev_require = [
    "pre-commit~=2.11",
]

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
        "pandas>=0.25.3,<1.2",
        "tensorflow>=2.4",
        "tensorflow-probability>=0.12",
        "pymatgen>=2020,<2020.9",
        "matminer>=0.6.2",
        "numpy>=1.18.3",
        "scikit-learn>=0.23,<0.24",
    ],
    tests_require=tests_require,
    test_suite="modnet.tests",
    extras_require={
        "test": tests_require,
        "dev": dev_require,
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
