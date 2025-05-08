def pivot_data(df):
    """
    Transforms eviction data into a pivot table format organized by location and date.
    
    This function processes eviction data by binning geographic coordinates, grouping by
    location and date, and creating a pivot table that shows eviction counts over time
    for each location. It also calculates the total number of evictions for each location.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing eviction data with at least the following columns:
        'Latitude', 'Longitude', 'Executed Date', and 'NTA' (Neighborhood Tabulation Area).
    
    Returns:
    --------
    tuple:
        - pivot_df : pandas.DataFrame
            A pivot table with locations (lat_lon, NTA) as index and dates as columns.
            Each cell contains the count of evictions for that location and date.
            Additional columns include 'latitude', 'longitude', and 'eviction_sum'.
        - df_geo : pandas.DataFrame
            Processed geographic data with binned coordinates and lat_lon pairs.
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
    return pivot_df, df_geo