"""
setup.py for Cortica

This file defines the package metadata and installation logic for Cortica,
a lightweight cognitive memory engine for semantic storage, decay-aware retrieval,
and conceptual traversal. Designed to be portable and plug-and-play, Cortica
can integrate with AI pipelines, RAG systems, or standalone reasoning tools.
"""

from setuptools import setup, find_packages

setup(
    name="cortica",
    version="0.1.0",
    description="A lightweight cognitive memory engine for semantic storage and retrieval.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nathan A. Lucy",
    license="MIT",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.10",
)
