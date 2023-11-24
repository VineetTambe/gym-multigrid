from setuptools import setup

setup(
    name="gym_multigrid",
    version="0.0.1",
    packages=["gym_multigrid", "gym_multigrid.envs"],
    install_requires=["gymnasium>=0.29.1", "numpy>=1.26.0"],
)
