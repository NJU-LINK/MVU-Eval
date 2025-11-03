# logging_utils.py
import os
import logging
from datetime import datetime


def setup_logging(data_dir, outfile_name_suffix):
    """配置日志系统"""

    logging.basicConfig(
        filename=os.path.join(data_dir, f'log_{outfile_name_suffix}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        # encoding='utf-8'
    )

    logging.info(f"Experiment setup: {outfile_name_suffix}")


def log_arguments(args):
    """记录命令行参数到日志"""
    args_dict = vars(args)
    logging.info("运行参数配置:\n")
    for k, v in args_dict.items():
        logging.info(f"{k}: {v}")
    logging.info("-"*60 + " 参数解析完成 " + "-"*60)
