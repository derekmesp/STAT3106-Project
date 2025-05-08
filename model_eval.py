import folium
from folium.plugins import HeatMapWithTime
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

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
    """
    Creates and displays a bar chart showing demographic distributions by eviction risk level.
    
    This function processes demographic data from the input DataFrame, converts percentage columns
    to numeric values, and generates a bar chart that visualizes the average demographic 
    composition for each eviction risk category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing demographic data with percentage columns and an 'Eviction Risk' column.
        Expected columns include '% Hispanic/Latino', '% White', '% Black', '%Asian', '% Other',
        and potentially '% 65+ Years' and '% 65+ Below Poverty'.
    
    Returns
    -------
    None
        Displays a matplotlib bar chart in the output.
    """
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
    
def graph_clusters(df, high_clusters, low_clusters):
    """
    Creates and displays a bar chart comparing demographic compositions between high and low eviction clusters.
    
    This function filters the input DataFrame for specified high and low eviction clusters,
    extracts demographic data, and generates a bar chart visualizing the average demographic
    composition for each cluster group.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing demographic data with a 'cluster' column and demographic percentage columns.
    high_clusters : list
        List of cluster identifiers representing areas with high eviction counts.
    low_clusters : list
        List of cluster identifiers representing areas with low eviction counts.
    
    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing two DataFrames:
        - high_demo: DataFrame with demographic data for high eviction clusters
        - low_demo: DataFrame with demographic data for low eviction clusters
    """
    high_demo = df[df['cluster'].isin(high_clusters)]
    low_demo = df[df['cluster'].isin(low_clusters)]
    demo_cols = ['% Hispanic/Latino', '% White', '% Black', '%Asian', '% Other']
    
    high_demo = high_demo[demo_cols]
    low_demo = low_demo[demo_cols]
    high_demo = high_demo.apply(pd.to_numeric, errors='coerce')
    low_demo = low_demo.apply(pd.to_numeric, errors='coerce')
    
    high_avg = high_demo.mean()
    low_avg = low_demo.mean()
    
    avg_df = pd.DataFrame({'Top 3 Clusters': high_avg, 'Bottom 3 Clusters': low_avg})
    avg_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Demographic Composition: Top 3 and Bottom 3 Clusters by Eviction Count') 
    plt.ylabel('Average Percentage')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    return high_demo, low_demo

def cluster_t_test(df_high, df_low):
    """
    Performs t-tests comparing demographic variables between high and low eviction clusters.
    
    This function conducts independent t-tests (assuming unequal variances) for each demographic
    variable between two groups of clusters. It also applies False Discovery Rate (FDR) correction
    using the Benjamini-Hochberg method to account for multiple comparisons.
    
    Parameters
    ----------
    df_high : pandas.DataFrame
        DataFrame containing demographic data for high eviction clusters.
        Each column represents a demographic variable.
    df_low : pandas.DataFrame
        DataFrame containing demographic data for low eviction clusters.
        Each column represents a demographic variable.
    
    Returns
    -------
    pandas.DataFrame
        A summary DataFrame containing the results of the t-tests with columns:
        - 'Demographic': The name of the demographic variable
        - 't-statistic': The t-statistic from the t-test
        - 'p-value': The p-value from the t-test
        - 'FDR-adjusted p-value': The p-value after FDR correction
        The DataFrame is sorted by adjusted p-values in ascending order.
        
    Raises
    ------
    ValueError
        If the input DataFrames don't have identical columns.
    """
    if not df_high.columns.equals(df_low.columns):
        raise ValueError("Dataframes must have the same columns")
    
    demo_cols = df_high.columns
    results = {
        'Demographic': [],
        't-statistic': [],
        'p-value': [],
        'FDR-adjusted p-value': []
    }

    # Run tests
    p_vals = []
    for col in demo_cols:
        h = df_high[col].dropna()
        l = df_low[col].dropna()
        
        if len(h) < 2 or len(l) < 2:
            t_stat = p = np.nan
        else:
            t_stat, p = ttest_ind(h, l, equal_var=False)
        
        results['Demographic'].append(col)
        results['t-statistic'].append(t_stat)
        results['p-value'].append(p)
        p_vals.append(p)

    # FDR correction
    _, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
    results['FDR-adjusted p-value'] = p_adj

    # Create DataFrame
    summary_df = pd.DataFrame(results)

    # Format nicely
    summary_df = summary_df.sort_values('FDR-adjusted p-value').reset_index(drop=True)
    summary_df['t-statistic'] = summary_df['t-statistic'].round(3)
    summary_df['p-value'] = summary_df['p-value'].apply(lambda p: f"{p:.6f}")
    summary_df['FDR-adjusted p-value'] = summary_df['FDR-adjusted p-value'].apply(lambda p: f"{p:.6f}")
    return summary_df