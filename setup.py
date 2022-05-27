from setuptools import find_packages, setup

setup(
    name="toolbox",
    packages=find_packages(),
    url="https://github.com/qgallouedec/toolbox",
    description="Some usefull tools",
    long_description=open("README.md").read(),
    install_requires=["numpy", "scipy", "rliable", "pygame", "Pillow"],
    extras_require={
        "extras": ["pytest", "black", "isort"],
    },
)
