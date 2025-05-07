import pandas as pd
import numpy as np
def load_eviction(data_path):
    """
    Load and preprocess eviction data from a CSV file.
    
    Parameters:
    data_path (str): The path to the CSV file containing the eviction data.
    
    Returns:
    pandas.DataFrame: A preprocessed DataFrame containing the eviction data.
    
    The function reads the CSV file using pandas, drops the 'Eviction Apartment Number' column,
    handles missing values by dropping rows with missing values, and removes duplicate entries.
    It also converts the 'Executed Date' column to datetime format and extracts the month and year.
    Finally, it prints the number of rows dropped due to missing values and duplicate entries,
    and the shape of the resulting DataFrame.
    """
    df = pd.read_csv(data_path)
    df = df.drop('Eviction Apartment Number', axis=1)
    print("{} rows were dropped due to missing values.".format(len(df) - len(df.dropna())))
    df = df.dropna()
    
    print("{} rows were dropped due to duplicate entries.".format(len(df) - len(df.drop_duplicates())))
    df = df.drop_duplicates()
    
    df['Executed Date'] = pd.to_datetime(df['Executed Date'])
    df['Month'] = df['Executed Date'].dt.month
    df['Year'] = df['Executed Date'].dt.year
    print("Eviction data with shape {} loaded.".format(df.shape))
    
    return df

def save_data(data_path, output_path):
    """
    Load eviction data, preprocess it, and save it to a CSV file.
    
    Parameters:
    data_path (str): The path to the input CSV file containing the raw eviction data.
    output_path (str): The path where the preprocessed data will be saved as a CSV file.
    
    Returns:
    None: The function does not return any value but saves the preprocessed data to the specified output path.
    """
    df = load_eviction(data_path)
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    data_path = 'Evictions_20250506.csv'
    output_path = 'preprocessed_eviction_data.csv'
    save_data(data_path, output_path)