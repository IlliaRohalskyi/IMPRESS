"""
Utility Module.

Provides utility functions

Example:
    from src.utils import get_project_root()
    project_root = get_project_root()
    print(f"The project root directory is: {project_root}")
"""
import os
import pickle


def get_project_root() -> str:
    """
    Get Project Root Directory.

    This function determines the root directory of a project based on the location of the script.
    It navigates upwards in the directory tree until it finds the setup.py file.

    Returns:
        str: The absolute path of the project's root directory.
    """
    script_path = os.path.abspath(__file__)

    # Navigate upwards in the directory tree until you find the setup.py file
    while not os.path.exists(os.path.join(script_path, "setup.py")):
        script_path = os.path.dirname(script_path)

    return script_path


def save_pickle(obj, file_path):
    """
    Save object as a pickle (pkl) file.

    Args:
        obj: The object to be saved.
        file_path (str): The path to the target pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    """
    Load data from a pickle (pkl) file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        The loaded object.
    """
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj
