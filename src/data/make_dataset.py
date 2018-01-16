"""
Script for applying the data processing tasks
"""
# -*- coding: utf-8 -*-
import os
import sys
import argparse

src_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'src'))
sys.path.append(src_dir)
import logging
import preprocessing



def str2bool(v):
    """ Argparse utility function for extracting boolean arguments.
    Original code is from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

FLAGS = None

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('start preprocessing data from raw:')
    if FLAGS.tweets:
        preprocessing.preprocess_tweets(save=True)
    else:
        preprocessing.preprocess()

    logger.info('files have been created in data/processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets',
                        type=str2bool,
                        default="True",
                        help='Set true, if twitter data should be processed.\
                              Make sure you have downloaded it before')
    FLAGS, unparsed = parser.parse_known_args()
    main()
