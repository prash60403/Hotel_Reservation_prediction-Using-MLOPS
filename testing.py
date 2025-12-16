from src.logger import logging
from src.custom_exception import CustomException
import sys

logger= logging.getLogger(__name__)

def divide_num(a,b):
    try:
        result= a/b
        logger.info(f"Division successful: {a} / {b} = {result}")
    except Exception as e:
        logger.error("Error occured during division")
        raise CustomException("Custom Error zero", sys)
    
if __name__ == "__main__":
    try:
        logger.info("Starting division operation")
        divide_num(10,0)
    except CustomException as ce:
        logger.error(str(ce))
