from setuptools import setup, find_packages

setup(
    name="nz_frontier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    author="Jinsu Park",
    author_email="jinsu.park@planit.institute",
    description="A Risk-Efficiency Theory of Corporate Decarbonization",
)
