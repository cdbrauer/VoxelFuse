import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="voxelfuse",
    version="1.0.0",
    author="Cole Brauer",
    description="A toolkit for processing 3D components made with mixtures of materials and multiple manufacturing processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdbrauer/VoxelFuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
)
