import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyroadacoustics",
    version="1.0.2",
    author="Stefano Damiano",
    author_email="stefano.damiano@esat.kuleuven.be",
    description="A package for simulating the sound propagation in a road scenario",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steDamiano/pyroadacoustics",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages = ["pyroadacoustics"],
    python_requires=">=3.7",
    package_data= {
        "" : ['*.json']
    },
    install_requires=[
        "numpy",
        "scipy>=0.18.0",
    ],
    keywords="outdoor aocustics automotive signal processing traffic sound simulation",
)