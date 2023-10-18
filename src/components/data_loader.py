"""
This module provides a function to load data from a DVC remote repository.
"""

import os
import subprocess


def load_dvc():
    """
    Loads data from a DVC remote repository.
    """
    username = os.environ.get("DVC_USERNAME")
    token = os.environ.get("DVC_TOKEN")

    subprocess.run(
        ["dvc", "remote", "modify", "origin", "--local", "auth", "basic"], check=True
    )
    subprocess.run(
        ["dvc", "remote", "modify", "origin", "--local", "user", username], check=True
    )
    subprocess.run(
        ["dvc", "remote", "modify", "origin", "--local", "password", token], check=True
    )

    subprocess.run(["dvc", "pull", "-r", "origin"], check=True)


if __name__ == "__main__":
    load_dvc()
