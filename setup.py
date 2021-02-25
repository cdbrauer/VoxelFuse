import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="voxelfuse",
    version="1.2.7",
    author="Cole Brauer",
    description="A toolkit for processing 3D components made with mixtures of materials and multiple manufacturing processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdbrauer/VoxelFuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    include_package_data=True,
)
