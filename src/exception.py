"""
Custom Exception Module.

Defines a custom exception class and a utility function for detailed error messages.
"""
import sys


def error_message_details(error, error_detail: sys):
    """
    Generate an error message with details.

    This function constructs an error message with information about the error,
    including the script name, line number, and error message.

    Args:
        error (str): The error message.
        error_detail (sys): The sys module detail of the error.

    Returns:
        str: The formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{filename}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{error}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception Class.

    Extends the built-in Exception class with enhanced error messages including script details.


    Args:
        error_message (str): The error message.
        error_detail (sys): The sys module detail of the error.

    Attributes:
        error_message (str): The formatted error message.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the CustomException instance.

        Args:
            error_message (str): The error message.
            error_detail (sys): The sys module detail of the error.
        """
        super().__init__(error_message)
        self.error_message = error_message_details(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """
        Return the custom error message.

        Returns:
            str: The formatted error message.
        """
        return self.error_message
