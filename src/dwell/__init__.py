"""Dwell: a stable local gateway for specialized AI runtimes."""

from importlib.metadata import version as distribution_version

__all__ = ["__version__"]

__version__ = distribution_version("dwell-ai")
