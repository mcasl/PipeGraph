"""Packaging shim.

All project metadata lives in ``pyproject.toml``. This file only keeps the
``verify`` command used by CircleCI to check that the git tag matches the
declared version before uploading a release.
"""
import os
import sys
import tomllib

from setuptools import setup
from setuptools.command.install import install

with open("pyproject.toml", "rb") as f:
    VERSION = tomllib.load(f)["project"]["version"]


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version."""
    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")
        if tag != VERSION:
            sys.exit(f"Git tag: {tag} does not match the version of this app: {VERSION}")


setup(cmdclass={"verify": VerifyVersionCommand})
