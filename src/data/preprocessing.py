"""
File for preprocessing routines on the fine managament data set
"""
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from data import data_utils
import json
import re
import nltk
from nltk.corpus import stopwords
import string


def preprocess(remove_NIL=False):
    data_dir = data_utils.get_data_dir()
    raw_data_dir = os.path.join(data_dir, "raw","road_traffic_fine_management_process.csv")
    data = pd.read_csv(raw_data_dir, sep=";")
    table_names = data["E:concept:name"].unique()
    for name in table_names:
        new_table = pd.DataFrame()
        concept_name_table = data[data["E:concept:name"]==name]
        new_table = pd.concat([new_table, concept_name_table])
        new_table = new_table.reset_index(drop=False)
        new_table = new_table.dropna(how="all", axis=1)
        new_table = new_table.loc[:, (new_table != 0).any(axis=0)]
        new_table = new_table.loc[:, (new_table != float(0)).any(axis=0)]
        if remove_NIL:
            new_table= new_table[new_table.apply(lambda x: sum([x_=='NIL' for x_ in x])==0, axis=1)]

        out_dir = os.path.join(data_dir,"processed")
        # make sure directory exists
        data_utils.setup_directory(out_dir)
        name_without_spaces = "_".join(name.split())
        out_dir = os.path.join(out_dir, str(name_without_spaces)+".csv")
        new_table.to_csv(out_dir, sep=";",index=False)

def clean_twitter_text(tweets):
    """
    This function removes common tags and letters in Tweets
    which create noise for the text mining task.
    The regex was taken from:
    https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
    and https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/
    it returns the tokenized tweets
    """
    emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

    regex_str = [
        emoticons_str,
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        r'(?:[\w_]+)', # other words
        r'(?:\S)' # anything else
    ]

    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

    tokenized_tweets = []
    for tweet_i in tweets:
        decoded_tweet = tweet_i.decode()
        tokens = tokens_re.findall(decoded_tweet)
        tokens = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(tokens)).split()).lower()
        tokenized_tweets.append(tokens)

    return tokenized_tweets

def _get_retweeted(tweets):
    """
    This method checks if the tweet was retweeted by the user in his/her
    timeline. retweeted texts have "rt" at the beginning of the text.
    """
    return [1 if tweet.startswith( 'rt', 0, 2 ) else 0
            for tweet in tweets]

def _remove_retweeted_tag(tweets):
    """
    This method removes the "rt" of retweeted texts at the beginning
    of a message.
    """
    cleaned_tweets = []
    for tweet in tweets:
        if tweet.startswith('rt', 0, 2 ):
            cleaned_tweets.append(tweet[3:])
        else:
            cleaned_tweets.append(tweet)
    return cleaned_tweets

def remove_stopwords(tweets, verbose=False):
    """
    This method removes typical stopwords from the english language
    and punctuation
    """
    nr_of_tweets = len(tweets)
    stop_word_list = stopwords.words('english') + list(string.punctuation) + ['via']
    cleaned_tweets = []
        if verbose:
            if i % 500 == 0:
                print("Process tweet number: {} of {} ".format(i,nr_of_tweets))
        cleaned_tweet_i = [term for term in tweet_i.split(" ") if term not in set(stop_word_list)]
        cleaned_tweet_i = ",".join(cleaned_tweet_i)
        cleaned_tweets.append(cleaned_tweet_i)
    return cleaned_tweets

def preprocess_tweets(file_path=None, columns_to_write=None, save=False, remove_stops=False):
    """
    This code provides a simple preprocessing routine to
    extract the needed data from a file of tweets saved in
    json format.
    """
    if file_path is None:
        data_dir = os.path.join(data_utils.get_data_dir(), "raw")
        file_path = os.path.join(
            data_dir, "realDonaldTrump", "realDonaldTrump_tweets.json")
    if not file_path.endswith(".json"):
        raise ValueError("Only json files are supported!")

    tweets = []
    with open(file_path, 'r') as f:
        for line in f:
            tweets = json.loads(line)

    if columns_to_write is None:
        colnames = ["id_str", "created_at", "full_text", "favorite_count", "retweet_count"]
    else:
        colnames = list(columns_to_write)

    extracted_tweets = []
    for tweet_i in tweets:
        tweet_extraction = [tweet_i[name].encode("utf-8") if (name == "full_text" or name == "text")  else tweet_i[name]
                            for name in colnames]
        extracted_tweets.append(tweet_extraction)

    # Convert date
    data = pd.DataFrame(extracted_tweets, columns=colnames)
    data["created_at"] = pd.to_datetime(data["created_at"])
    data["full_text"] = clean_twitter_text(data["full_text"])
    data["retweeted"] = _get_retweeted(data["full_text"])
    data["full_text"] = _remove_retweeted_tag(data["full_text"])
    if remove_stops:
        data["full_text"] = remove_stopwords(data["full_text"])

    # remove nan values and empty strings
    data = data.replace('', np.nan)
    data = data.dropna()



    if save:
        # write csv file
        data_dir = os.path.join(data_utils.get_data_dir(), "processed")
        _, file_name = os.path.split(file_path)
        file_name = file_name[:-5] + ".csv"
        out_dir = os.path.join(data_dir, file_name)
        data.to_csv(out_dir, sep=";", index=False)

    return data


########
#
# Maybe need them later
#
########

def convert_timestamps_from_dataframe(df, time_col_names, unit="ms" ):
    """
    This method converts integer timestamp columns in a pandas.DataFrame object
    to datetime objects.
    DataFrames in mobility data with datetime columns:
    cell, event, location, marker, sensor

    Parameters
    ----------
    data: input data pandas DataFrame.
    unit: string, default="ms"
          time unit for transformation of the input integer timestamps.
          Possible values: D,s,ms,us,ns
          see "unit" at: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html
          for further information.

    Returns
    -------
    result: returns a deepcopy of the data with transformed time columns.
    """

    df_column_names = list(df.columns.values)
    if any(name_i in time_col_names for name_i in df_column_names):
        result = deepcopy(df)
        for time_column_name in time_col_names:
            if time_column_name in df_column_names:
                index_copy = result.index
                result.set_index(time_column_name,inplace=True)
                result.index = pd.to_datetime(result.index, unit=unit)
                result.reset_index(inplace=True)
                result.index = index_copy
    return result

def downsample_time_series(series, time_interval="S", time_col_name="time"):
    """
    Downsamples a pandas time series DataFrame from milliseconds to a new
    user specified time interval. The aggregation for the new time bins will be
    calculated via the mean. To make sure that the right time column is
    used you have to set the time columns name in time_col_name or set it as
    index before calling this function.
    Otherwise it is assumed that the time column has the name time_col_name="time".

    For further information about examples for pandas resampling function see:
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling
    https://machinelearningmastery.com/resample-interpolate-time-series-data-python/


    Parameters
    ----------
    series: a pandas DataFrame object with a DatetimeIndex, if there is no
        DatetimeIndex set, it is assumed that there is a Datetime column with
        name time_col_name="time"
    time_interval: string, default="S",
        specifies the new time interval to which the series will be downsampled.
        Valid values are "S" for seconds, "T" for minutes etc. It is also possible
        to sample in a special interval e.g. 5 seconds, by passing "5S".
        For all possible frequencies see:
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation#17001474

    time_col_name: string, default="time"
        The name of the time column name.
    Returns
    -------
    data: returns the data with downsampled time columns, where each new bin
          is aggregated via the mean.
    """

    if isinstance(series.index, pd.DatetimeIndex):
        resampled = series.resample(time_interval).mean()
    elif time_col_name in list(series.columns.values):
        # In case the column has not been converted to Datetime object
        # it will be converted here.
        if series[time_col_name].dtype in [np.dtype("Int64")]:
            series = deepcopy(series)
            series = convert_timestamps(series, time_col_names=[time_col_name])
        resampled = series.set_index(time_col_name).resample(time_interval).mean()
        resampled = resampled.reset_index()
    else:
        resampled = series
    return resampled

def downsample_time_series_per_category(series, categorical_colnames, time_interval="S", time_col_name="time"):
    """
    Downsamples a pandas time series DataFrame from milliseconds to a new
    user specified time interval and takes care of the right interpolation of categorical variables.
    The aggregation for the new time bins will be calculated via the mean.
    To make sure that the right time column is used you have to set the time
    columns name in time_col_name.
    Otherwise it is assumed that the time column has the name time_col_name="time".

    For further information about examples for pandas resampling function see:
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling
    https://machinelearningmastery.com/resample-interpolate-time-series-data-python/


    Parameters
    ----------
    series: a pandas DataFrame object with a DatetimeIndex, if there is no
        DatetimeIndex set, it is assumed that there is a Datetime column with
        name time_col_name="time"
    categorical_colnames: a list of strings of colum names
        e.g. ["sensor"]
    time_interval: string, default="S",
        specifies the new time interval to which the series will be downsampled.
        Valid values are "S" for seconds, "T" for minutes etc. It is also possible
        to sample in a special interval e.g. 5 seconds, by passing "5S".
        For all possible frequencies see:
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation#17001474
    time_col_name: string, default="time"
        The name of the time column name. set to "index" if you want to transform
        the index column
    Returns
    -------
    data: returns the data with downsampled time columns, where each new bin
          is aggregated via the mean and keeps the categorical values.
    """
    series_column_names = list(series.columns.values)
    result = pd.DataFrame(columns = series_column_names)
    # In case the column or index has not been converted to Datetime object
    # it will be converted here.
    if (time_col_name=="index") and (series.index.dtype in [np.dtype("Int64")]):
        series = deepcopy(series)
        series.index = pd.to_datetime(series.index, unit="ms")
    if time_col_name in series_column_names:
        if series[time_col_name].dtype in [np.dtype("Int64")]:
            series = deepcopy(series)
            series = _convert_timestamps_from_dataframe(series, time_col_names=[time_col_name])

    # Start actual downsampling
    if isinstance(series.index, pd.DatetimeIndex) or (time_col_name in series_column_names):
        for categorical_colname_i in categorical_colnames:
            categories = list(series[categorical_colname_i].unique())
            for category_i in categories:
                series_for_category = series[series[categorical_colname_i]==category_i]
                resampled = downsample_time_series(series_for_category, time_interval, time_col_name)
                resampled[categorical_colname_i] = category_i
                result = pd.concat([result, resampled])

        if isinstance(result.index, pd.DatetimeIndex):
            result = result.sort_index()
        else:
            result = result.set_index(time_col_name).sort_index()
        # need to reset index otherwise indices could be not unique anymore
        result = result.reset_index()
    else:
        result = series
    return result
