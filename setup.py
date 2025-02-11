from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Equivolumetric layering"
LONG_DESCRIPTION = (
    "A package to perform equivolumetric layering on images with minimal dependencies."
)

setup(
    name="equivol",
    version=VERSION,
    author="Matthew Sutton",
    author_email="sutton.matt.p@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-image",
    ],
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
