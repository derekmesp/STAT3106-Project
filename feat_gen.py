from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def feature_generation(df):
    """
    Transforms raw eviction data into engineered features for analysis or modeling.
    
    This function processes a dataframe containing eviction data by:
    1. Removing non-feature columns
    2. Binning and standardizing geographic coordinates
    3. Creating frequency features for addresses, neighborhoods, and marshals
    4. Converting categorical variables to binary indicators
    5. Separating metadata from features used for modeling
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing raw eviction data with columns including 
        'Latitude', 'Longitude', 'Eviction Address', 'Marshal First Name', 
        'Marshal Last Name', 'Ejectment', 'Residential/Commercial', 'NTA',
        'Year', and 'Eviction/Legal Possession'.
        
    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        A tuple containing:
        - metadata: DataFrame with location and identification columns
        - X: DataFrame with engineered features suitable for analysis or 
          machine learning, with original coordinates replaced by standardized bins,
          frequency features, and binary encoded categorical variables
    """
    exclude = ['Court Index Number', 'Docket Number '] 
    features = [col for col in df.columns if col not in exclude]
    df_features = df[features]
    df_features = df_features.copy()
    
    #make community board, council district, 

    df_features['lat_bin'] = df_features['Latitude'].round(3)
    df_features['lon_bin'] = df_features['Longitude'].round(3)
    df_features = df_features.drop(['Latitude', 'Longitude'], axis=1)

    scaler = StandardScaler()
    df_features[['lat_bin', 'lon_bin']] = scaler.fit_transform(df_features[['lat_bin', 'lon_bin']])
    
    address_counts = df_features['Eviction Address'].value_counts()
    df_features['AddressFreq'] = df['Eviction Address'].map(address_counts)
    
    scaler = MinMaxScaler()
    df_features['AddressFreq'] = scaler.fit_transform(df_features[['AddressFreq']])
    df_features['Year'] = scaler.fit_transform(df_features[['Year']])
    
    nta_counts = df['NTA'].value_counts()
    df_features['NeighborhoodEvictionFreq'] = df_features['NTA'].map(nta_counts)
    df_features['NeighborhoodEvictionFreq'] = scaler.fit_transform(df_features[['NeighborhoodEvictionFreq']])
    
    df_features['Marshal'] = df_features['Marshal First Name'] + ' ' + df['Marshal Last Name']
    marshal_freq = df_features['Marshal'].value_counts()
    df_features['MarshalActivity'] = df_features['Marshal'].map(marshal_freq)
    df_features = df_features.drop(['Marshal First Name', 'Marshal Last Name'], axis=1)
    
    df_features['MarshalActivity'] = np.log1p(df_features['MarshalActivity'])
    
    df_features['Ejectment'] = df_features['Ejectment'].apply(lambda x: 0 if x == 'Not an Ejectment' else 1)
    df_features['Residential/Commercial'] = df_features['Residential/Commercial'].apply(lambda x: 0 if x == 'Residential' else 1)
    df_features['Eviction/Legal Possession'] = df_features['Eviction/Legal Possession'].apply(lambda x: 0 if x == 'Eviction' else 1)
    
    exclude = ['Eviction Address', 'Executed Date', 'BOROUGH', 'Eviction Postcode', 'Community Board', 'Council District',
       'Census Tract', 'BIN', 'BBL', 'NTA', 'Marshal']
    
    valid_exclude = [col for col in exclude if col in df_features.columns]
    metadata = df_features[valid_exclude]
    
    X = df_features.drop(exclude, axis=1)
    return metadata, X