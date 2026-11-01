from setuptools import setup, find_packages

setup(
    name="torch-morph-rcc8",
    version="0.4.0",
    description="Differentiable Fuzzy RCC-8 Spatial Logic & Max-Tree Morphological Operators in PyTorch",
    author="Jinwon Kim",
    url="https://github.com/Gelukkig95/torch-morph-rcc8",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "higra",
        "shapely",
        "scikit-image",
    ],
)
