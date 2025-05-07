import pandas as pd
import numpy as np
def load_eviction(data_path):
    """
    Load and preprocess eviction data from a CSV file.
    
    This function reads eviction data from a CSV file, removes the 'Eviction Apartment Number' column,
    drops rows with missing values and duplicate entries, converts the 'Executed Date' column to datetime format,
    and extracts the year from the date into a new 'Year' column. It also converts several numeric ID columns
    to string format for consistency.
    
    Parameters:
    data_path (str): The file path to the CSV file containing the eviction data.
    
    Returns:
    pandas.DataFrame: A preprocessed DataFrame containing the cleaned eviction data with appropriate data types.
    """
    df = pd.read_csv(data_path)
    df = df.drop('Eviction Apartment Number', axis=1)
    print("{} rows were dropped due to missing values.".format(len(df) - len(df.dropna())))
    df = df.dropna()
    
    print("{} rows were dropped due to duplicate entries.".format(len(df) - len(df.drop_duplicates())))
    df = df.drop_duplicates()
    
    df['Executed Date'] = pd.to_datetime(df['Executed Date'])
    df['Year'] = df['Executed Date'].dt.year
    print("Eviction data with shape {} loaded.".format(df.shape))
    
    df['Community Board'] = df['Community Board'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
    df['Council District'] = df['Council District'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
    df['Census Tract'] = df['Census Tract'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
    df['BIN'] = df['BIN'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
    df['BBL'] = df['BBL'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
    

    print("After conversion:")
    print(df.dtypes)
    
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