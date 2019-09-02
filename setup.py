import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="voxelfuse",
    version="1.0.0",
    author="Cole Brauer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Team-Automata/VoxelFuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
)
