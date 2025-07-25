import logging
import os

def setup_logger():
    pid = os.getpid()
    logger = logging.getLogger(f"Process {pid}")
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    
    # Read dir form temp_pathname.txt file
    dir = open("temp_pathname.txt", "r").read().split("\n")[0].split("/")[-1]

    if not os.path.exists(f"logs/{dir}"):
        os.mkdir(f"logs/{dir}")

    # Avoid adding multiple file handlers if logger is already set up
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(f"logs/{dir}/{pid}.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger