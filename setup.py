import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyroadacoustics",
    version="0.0.1",
    author="Stefano Damiano",
    author_email="stefano.damiano@esat.kuleuven.be",
    description="A package for simulating the sound propagation on a road scenario",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steDamiano/outdoorSimulator",
    classifiers=[
        "Development Status :: 3-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "License :: To be defined :: To be defined",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "src"},
    packages = ["pyroadacoustics"],
    package_data={"pyroadacoustics": ["materials.json"]},
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy>=0.18.0",
    ],
    keywords="outdoor aocustics automotive signal processing traffic sound simulation",
)