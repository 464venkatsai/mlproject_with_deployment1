import sys
from src.logger import logging

def error_msg(error,error_detail:sys):
    i,j,exe_tb = error_detail.exc_info()
    # print(i,j)
    file_name = exe_tb.tb_frame.f_code.co_filename
    error_msg = f'Error in  script name {file_name} in line number {exe_tb.tb_lineno} error message {str(error)}'
    return error_msg
    
class CustomException (Exception):
    def __init__(self,error_message,error_details: sys):
        super().__init__(error_message)
        self.error_message = error_msg(error_message,error_detail=error_details)
        
    def __str__(self):
        return self.error_message

        