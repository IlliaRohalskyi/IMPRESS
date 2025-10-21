"""
Setup Module.

This module provides a setup configuration for the ReProTENSID project.

The setup configuration includes package details, version, and installation requirements.
"""
from typing import List

from setuptools import find_packages, setup


def get_requirements(filepath: str) -> List[str]:
    """
    Get Project Requirements.

    This function reads and returns a list of project requirements from a file.

    Args:
        filepath (str): The path to the requirements file.

    Returns:
        List[str]: A list of project requirements.
    """
    requirements = []
    with open(filepath, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="ReProTENSID",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
