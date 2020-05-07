import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
    
setuptools.setup(
    name="modnet",
    version="0.1.1",
    author="Pierre-Paul De Breuck",
    author_email="pierre-paul.debreuck@uclouvain.be",
    description="MODNet, the Material Optimal Descriptor Network for materials properties prediction. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ppdebreuck/modnet",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
          'pandas>=0.25.3',
          'keras>=2.3',
          'pymatgen>=2020.3.13',
          'matminer>=0.6.2',
          'numpy>=1.18.3',
          'scikit-learn'
        ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
