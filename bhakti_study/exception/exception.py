import sys
import logging

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)
        logging.error(self.error_message)  # Logs the error message

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        _, _, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            exception_block_line_number = exc_tb.tb_frame.f_lineno
            try_block_line_number = exc_tb.tb_lineno

            return f"""
            Error occurred in script: [{file_name}]
            Try block line number: [{try_block_line_number}]
            Exception block line number: [{exception_block_line_number}]
            Error message: [{error_message}]
            """
        return str(error_message)

    def __str__(self):
        return self.error_message
