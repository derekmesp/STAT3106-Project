import folium
from folium.plugins import HeatMapWithTime
from IPython.display import display
from matplotlib import pyplot as plt
import pandas as pd

def cluster_time_series(df, clusters, radius=8, auto_play=True, max_opacity=0.8):
    """
    Creates and displays an interactive time series heatmap of specified clusters.
    
    This function filters the input dataframe for the specified clusters, then creates
    a time series heatmap showing the geographical distribution of data points over time.
    The map is centered on New York City coordinates.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data with columns 'cluster', 'Executed Date', 'lat_bin', and 'lon_bin'.
    clusters : list
        List of cluster identifiers to include in the visualization.
    radius : int, optional
        Radius of each data point on the heatmap in pixels (default is 8).
    auto_play : bool, optional
        Whether the time series animation should play automatically (default is True).
    max_opacity : float, optional
        Maximum opacity value for the heatmap points, between 0 and 1 (default is 0.8).
    
    Returns
    -------
    None
        Displays the interactive folium map in the output.
    """
    filtered_df = df[df['cluster'].isin(clusters)]

    dates = sorted(filtered_df["Executed Date"].unique())
    heat_data = [
        filtered_df[filtered_df["Executed Date"] == date][["lat_bin", "lon_bin"]].values.tolist()
        for date in dates
    ]
    top_nta = filtered_df.groupby('cluster')['NTA'].agg(lambda x: x.value_counts().head(3).index.tolist()).reset_index()
    print(top_nta)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    HeatMapWithTime(
            data=heat_data,
            index=dates,
                radius=radius,
                auto_play=auto_play,
                max_opacity=max_opacity
        ).add_to(m)
    display(m)
    
def graph_demographic(df):
    columns_to_convert = ['% 65+ Years', '% 65+ Below Poverty', '% Hispanic/Latino', '% White', '% Black', '%Asian', '% Other']
    
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    columns = ['% Hispanic/Latino', '% White', '% Black', '%Asian', '% Other', 'Eviction Risk']
    demographics = df[columns]
    average_demographics = demographics.groupby('Eviction Risk').mean()
    average_demographics.plot(kind='bar', figsize=(10, 6))

    plt.title('Demographics by Eviction Risk Level')
    plt.xlabel('Eviction Risk')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    