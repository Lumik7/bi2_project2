"""
Script for applying the data processing tasks
"""



# -*- coding: utf-8 -*-
import os
import logging
import preprocessing



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('start preprocessing data from raw:')
    preprocessing.preprocess()
    logger.info('files have been created in data/processed')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
