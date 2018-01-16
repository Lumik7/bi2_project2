"""
File for downloading tweets from twitter.com
The code has been adapted from:
https://www.kdnuggets.com/2016/06/mining-twitter-data-python-part-1.html
and http://docs.tweepy.org/en/v3.5.0/api.html
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv
import json

import tweepy
from tweepy import OAuthHandler
import sys
src_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'src'))
sys.path.append(src_dir)
from data import data_utils

def store_json(tweet, name, append=False):
    """
    This function stores the tweet as json file
    """
    mode = "w"
    if append:
        mode = "a"
    with open(str(name), mode) as f:
        #json.dump(tweet, f, indent=4, sort_keys=True)
        json.dump(tweet, f)

FLAGS = None

def main():
    logger = logging.getLogger(__name__)

    # Load authentification data from .env file
    load_dotenv(find_dotenv())
    consumer_key = os.environ.get('YOUR-CONSUMER-KEY')
    consumer_secret = os.environ.get('YOUR-CONSUMER-SECRET')
    access_token = os.environ.get('YOUR-ACCESS-TOKEN')
    access_secret = os.environ.get('YOUR-ACCESS-SECRET')

    # setup tweet connection
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    nr_of_tweets = int(FLAGS.count)
    max_allowed_count = 200
    if nr_of_tweets < max_allowed_count:
        max_allowed_count = nr_of_tweets
        nr_of_iterations = 0
    else:
        nr_of_iterations = int(round(nr_of_tweets // float(max_allowed_count)))
    all_tweets = []

    logger.info('start downloading tweets from {}'.format(FLAGS.user_name))

    # initialize tweet list to get a valid last tweet id
    latest_tweets = api.user_timeline(screen_name=FLAGS.user_name, tweet_mode="extended", count=max_allowed_count)
    all_tweets.extend(latest_tweets)
    last_tweet_id = all_tweets[-1].id - 1

    for batch_i in range(nr_of_iterations):
        latest_tweets = api.user_timeline(screen_name=FLAGS.user_name, tweet_mode="extended", count=max_allowed_count, max_id=last_tweet_id )
        all_tweets.extend(latest_tweets)
        last_tweet_id = all_tweets[-1].id - 1

    logger.info('finished downloading')

    data_dir = data_utils.get_data_dir()
    user_dir = os.path.join(data_dir,"raw",FLAGS.user_name)
    data_utils.setup_directory(user_dir)
    file_path = os.path.join(user_dir, FLAGS.user_name + "_tweets.json")

    logger.info('start writing tweets to json')

    all_tweets = [tweet._json for tweet in all_tweets]
    with open(file_path, "w") as f:
        json.dump(all_tweets, f)

    logger.info('finished writing tweets to json')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_name',
                        type=str,
                        default="realDonaldTrump",
                        help='specify from which user public tweets should be downloaded, \
                              note that twitter user naems are case sensitive.')
    parser.add_argument('--count',
                        type=int,
                        default=20,
                        help='specify the number of tweets which should be downloaded')

    FLAGS, unparsed = parser.parse_known_args()
    main()
