from sklearn.mixture import BayesianGaussianMixture

def feature_processing(df):
    """
    Process dataframe to create weighted representation of eviction data based on geographic coordinates.
    
    This function transforms the input dataframe by creating a weight column from eviction counts,
    filtering for positive eviction weights, and replicating rows based on their eviction weight.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at minimum 'eviction_sum', 'longitude', and 'latitude' columns.
        
    Returns
    -------
    tuple
        A tuple containing:
        - X_rep : numpy.ndarray
            Array of longitude and latitude values, replicated according to eviction weights.
        - replicated_df : pandas.DataFrame
            Dataframe with rows replicated according to their eviction weights.
    """
    df['eviction_weight'] = df['eviction_sum'].round().clip(upper=20).astype(int)
    weighted_df = df[df['eviction_weight'] > 0].copy()
    replicated_df = weighted_df.loc[weighted_df.index.repeat(weighted_df['eviction_weight'])]
    X_rep = replicated_df[['latitude', 'longitude']].values
    return X_rep, replicated_df
    
def create_bgmm(df):
    X_rep, replicated_df = feature_processing(df)
    bgmm = BayesianGaussianMixture(n_components=30, random_state=0)
    replicated_df['cluster'] = bgmm.fit_predict(X_rep)
    cluster_map = replicated_df.groupby(replicated_df.index)['cluster'].agg(lambda x: x.value_counts().idxmax())
    return bgmm, cluster_map