#!/usr/bin/env python3
from setuptools import setup
    
setup(
    name="hear_nt",
    packages=["hear_nt"],
    version="0.0.1",
    description="HEAR 2021",
    python_requires=">=3.7",
    install_requires=[
        "librosa==0.8.1",
        "numba==0.48",
        "numpy==1.19.2",
        "tensorflow==2.6.0",        
        "numba==0.48",
        # "numba>=0.49.0", # not directly required, pinned by Snyk to avoid a vulnerability
        "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
        "hearbaseline",
        "gdown==4.2.0"
    ]
)
