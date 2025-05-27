"""Cortica Setup.

This file defines the package metadata and installation logic for Cortica,
a lightweight cognitive memory engine for semantic storage, decay-aware retrieval,
and conceptual traversal. Designed to be portable and plug-and-play, Cortica
can integrate with AI pipelines, RAG systems, or standalone reasoning tools.
"""

import os
import re
from setuptools import find_packages, setup


def read_version():
    with open(os.path.join("cortica", "version.py")) as f:
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', f.read())
        if not match:
            raise RuntimeError("Version string not found.")
        return match.group(1)


setup(
    name="cortica",
    version=read_version(),
    description="A lightweight cognitive memory engine for semantic storage and retrieval.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nathan A. Lucy",
    license="MIT",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.10",
)
