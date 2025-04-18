import os

import setuptools

from smac import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        "setuptools",
        "types-setuptools",
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl-sphinx-theme>=0.2",
        # Others
        "mypy",
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        "pre-commit",
        "pylint",
        "mkdocs",
        "mkdocs-material",
        "mkdocs-autorefs",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-glightbox",
        "mkdocs-glossary-plugin",
        "mkdocstrings[python]",
        "markdown-exec[ansi]",
        "mike",
        "pillow",
        "cairosvg",
        "black",                # This allows mkdocstrings to format signatures in the docs
        "pytest",
        "pytest-coverage",
        "pytest-cases",
    ],
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="BSD 3-Clause License",
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy>=1.9.2",
        "psutil",
        "pynisher>=1.0.0",
        "ConfigSpace>=1.0.0",
        "joblib",
        "scikit-learn>=1.1.2",
        "pyrfr>=0.9.0",
        "dask[distributed]",
        "dask_jobqueue>=0.8.2",
        "emcee>=3.0.0",
        "regex",
        "pyyaml",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
