import setuptools
import re

with open("README.md", "r") as f:
    long_description = f.read()

with open("modnet/__init__.py", "r") as f:
    lines = ""
    for item in f.readlines():
        lines += item + "\n"


version = re.search('__version__ = "(.*)"', lines).group(1)

setuptools.setup(
    name="modnet",
    version=version,
    author="Pierre-Paul De Breuck",
    author_email="pierre-paul.debreuck@uclouvain.be",
    description="MODNet, the Material Optimal Descriptor Network for materials properties prediction. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ppdebreuck/modnet",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[],
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
