# setup.py
from setuptools import setup, find_packages

setup(
    name="xray_tools",
    version="0.1.0",
    description="SAXS/WAXS data analysis tools",
    packages=find_packages(where="tools"),
    package_dir={"": "tools"},
    install_requires=[
        "numpy",
        "scipy", 
        "matplotlib",
        "pandas",
        "fabio",
        "jupyter"
    ],
    python_requires=">=3.7",
)