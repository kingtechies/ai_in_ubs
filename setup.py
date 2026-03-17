"""
TinyFormer-USB – USB AI accelerator reference implementation.
"""

from setuptools import find_packages, setup

setup(
    name="tinyformer_usb",
    version="0.1.0",
    description="USB-form-factor AI inference accelerator: 20 kt/s on a USB stick",
    packages=find_packages(exclude=["tests*", "benchmarks*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
        "usb": ["pyusb>=1.2.1"],
    },
)
