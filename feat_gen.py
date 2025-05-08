def pivot_data(df):
    """
    This function takes a DataFrame containing eviction data and performs data preprocessing and feature generation.
    It groups the data by latitude, longitude, NTA (Neighborhood Tabulation Area), and executed date,
    then pivots the data to create a time series representation of evictions per NTA.
    Finally, it calculates the sum of evictions for each NTA.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing eviction data with columns 'Latitude', 'Longitude', 'Executed Date', and 'NTA'.

    Returns:
    pivot_df (pandas.DataFrame): A DataFrame with the following columns:
        - 'lat_lon': A tuple of latitude and longitude rounded to 3 decimal places.
        - 'NTA': Neighborhood Tabulation Area.
        - 'Executed Date': Date of eviction.
        - 'count': Count of evictions for each combination of 'lat_lon', 'NTA', and 'Executed Date'.
        - 'latitude': Latitude rounded to 3 decimal places.
        - 'longitude': Longitude rounded to 3 decimal places.
        - 'eviction_sum': Sum of evictions for each NTA.
    """
    df_geo = df[['Latitude', 'Longitude', 'Executed Date', 'NTA']]
    df_geo = df_geo.copy()  
    df_geo['lat_bin'] = df_geo['Latitude'].round(3)
    df_geo['lon_bin'] = df_geo['Longitude'].round(3)
    df_geo = df_geo.drop(['Latitude', 'Longitude'], axis=1)
    
    df_geo['lat_lon'] = list(zip(df_geo['lat_bin'], df_geo['lon_bin']))
    df_geo = df_geo.sort_values('Executed Date')
    
    grouped = (
        df_geo
        .groupby(['lat_lon', 'NTA', 'Executed Date'])
        .size()
        .reset_index(name='count')
    )
    pivot_df = grouped.pivot_table(
        index=['lat_lon', 'NTA'],
        columns='Executed Date',
        values='count',
        fill_value=0
    )
    
    pivot_df = pivot_df.drop_duplicates()
    pivot_df = pivot_df.sort_values(by=['lat_lon'])
    pivot_df['latitude'] = pivot_df.index.get_level_values(0).map(lambda x: x[0])
    pivot_df['longitude'] = pivot_df.index.get_level_values(0).map(lambda x: x[1])
    
    non_date_columns = ['NTA', 'latitude', 'longitude']
    date_columns = [col for col in pivot_df.columns if col not in non_date_columns]
    pivot_df['eviction_sum'] = pivot_df[date_columns].sum(axis=1) 
    return pivot_df