import pandas as pd
import matplotlib.pyplot as plt
import folium 
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
from IPython.display import display

class EvictionVisualizer:
    def __init__(self, df):
        """
        Initialize the EvictionVisualizer with a copy of the eviction DataFrame.
        """
        self._df = df.copy()
        self._df_geo = None
        self._eviction_counts = None

    def _compute_eviction_counts(self):
        """
        Private method to calculate eviction counts by borough and year.
        """
        self._eviction_counts = (
            self._df.groupby(['BOROUGH', 'Year'])
            .size()
            .unstack(fill_value=0)
        )

    def get_eviction_counts(self):
        """
        Public method to return the borough-year eviction count DataFrame.
        """
        if self._eviction_counts is None:
            self._compute_eviction_counts()
        return self._eviction_counts

    def plot_evictions_over_time(self):
        """
        Generates a line plot of evictions per borough over time.
        """
        eviction_counts = self.get_eviction_counts()
        eviction_counts.T.plot(figsize=(12, 6))
        plt.title("Evictions per Borough by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Evictions")
        plt.legend(title="Borough")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compute_geo(self):
        """
        Processes and stores geographic data with rounded lat/lon and sorted by year.

        This function filters and bins geographic coordinates (lat/lon) from the dataset
        and stores it for time-based geographic plotting.
        """
        df_geo = self._df[["Latitude", "Longitude", "Year"]].copy()
        df_geo["lat_bin"] = df_geo["Latitude"].round(3)
        df_geo["lon_bin"] = df_geo["Longitude"].round(3)
        df_geo = df_geo.sort_values("Year")
        self._df_geo = df_geo

    def get_geo_df(self):
        """
        Public method to access the processed geographic DataFrame.
        
        Returns
        -------
        pandas.DataFrame or None
            The processed geographic DataFrame with lat/lon bins, or None if not computed.
        """
        if self._df_geo is None:
            self.compute_geo()
        return self._df_geo
    def plot_time_series_heatmap(self, radius=8, auto_play=True, max_opacity=0.8):
        """
        Plots a time-series heatmap of evictions using HeatMapWithTime.

        Parameters
        ----------
        radius : int, optional
            Radius of each point in the heatmap. Default is 8.
        auto_play : bool, optional
            Whether the heatmap animation plays automatically. Default is True.
        max_opacity : float, optional
            Maximum opacity of heatmap points. Default is 0.8.

        Returns
        -------
        folium.Map
            A folium map object with the animated heatmap added.
        """
        if self._df_geo is None:
            self.compute_geo()
        geo = self._df_geo
        years = sorted(geo["Year"].unique())
        heat_data = [
            geo[geo["Year"] == year][["lat_bin", "lon_bin"]].values.tolist()
            for year in years
        ]

        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        HeatMapWithTime(
            data=heat_data,
            index=years,
            radius=radius,
            auto_play=auto_play,
            max_opacity=max_opacity
        ).add_to(m)
        display(m)
    def plot_static_heatmap(self):
        """
        Plots a static heatmap of evictions using HeatMap.
        Returns
        -------
        folium.Map
        A folium map object with the heatmap added.
        """
        if self._df_geo is None:
            self.compute_geo()
        geo = self._df_geo
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        if self._df_geo is None:
            self.compute_geo()
        geo = self._df_geo
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        heat_data = geo[["lat_bin", "lon_bin"]].values.tolist()
        HeatMap(heat_data, radius=10).add_to(m)
        display(m)

 
                