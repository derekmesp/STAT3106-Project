from sklearn.mixture import BayesianGaussianMixture

def feature_processing(df):
    df['eviction_weight'] = df['eviction_sum'].round().clip(upper=20).astype(int)
    weighted_df = df[df['eviction_weight'] > 0].copy()
    replicated_df = weighted_df.loc[weighted_df.index.repeat(weighted_df['eviction_weight'])]
    X_rep = replicated_df[['longitude', 'latitude']].values
    return X_rep, replicated_df
    
def create_bgmm(df):
    X_rep, replicated_df = feature_processing(df)
    bgmm = BayesianGaussianMixture(n_components=30, random_state=0)
    replicated_df['cluster'] = bgmm.fit_predict(X_rep)
    cluster_map = replicated_df.groupby(replicated_df.index)['cluster'].agg(lambda x: x.value_counts().idxmax())
    return bgmm, cluster_map