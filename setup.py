from setuptools import setup, find_packages

setup(
    name="CHMPredict",
    version="0.1.0",
    author="@nis",
    license="GPLv3+",
    packages=find_packages(include=["chmpredict"]),
    install_requires=[
        'tqdm',
        'h5py',
        'pillow',
        'rasterio',
        'torchvision',
        'scikit-learn',
        'configargparse',
    ],
)