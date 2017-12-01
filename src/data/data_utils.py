import os
import pandas as pd
import numpy as np
import tarfile
import preprocessing

def setup_directory(dir_name):
    """Setup directory in case it does not exist
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Created Directory: {}".format(dir_name) )
        except:
            print("Could not create directory: {}".format(dir_name))


def get_data_dir():
    """ Returns the data dir relative from this file
    """
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    data_dir = os.path.join(project_dir,"data")
    return data_dir

def _get_file_names(dir_name, file_ending=".csv"):
    """ Returns all file names from specified directory
    """
    file_names = list()
    dir_name = os.path.join(get_data_dir(),dir_name)
    if os.path.exists(dir_name):
        dir_content = os.listdir(dir_name)
        for file_name in dir_content:
            if file_name.endswith(file_ending):
                file_names.append(file_name)
    return file_names

def read_csv_from_dir(dir_name="raw"):
    """
    This function reads all .csv files from the specified
    directory in the data directory e.g. if dir_name = "raw" it reads
    all csv files from data/raw/. All read csv files are appended
    in a list and then returned. The csv tables are stored alphabetically.

    Parameters
    ----------
    dir_name: name of the directory inside the data directory.
        e.g. "raw", "processed", "interim" or "external"
    Returns
    -------
    csv_dfs: a list of pandas.DataFrame's
    """
    file_names = _get_file_names(dir_name=dir_name)
    data_dir = os.path.join(get_data_dir(), dir_name)
    csv_dfs = []
    for file_name_i in file_names:
        path_i = os.path.join(data_dir,file_name_i)
        df = pd.read_csv(path_i, sep=";")
        csv_dfs.append(df)

    return csv_dfs
