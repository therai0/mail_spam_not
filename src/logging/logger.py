import logging
import os
from datetime import datetime
from tarfile import data_filter

log_file_name = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
log_dir_path = os.path.join(os.getcwd(), "logs", log_file_name)
os.makedirs(log_dir_path)

log_file_path = os.path.join(log_dir_path, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
