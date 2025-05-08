from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline

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
    """
    Create a Bayesian Gaussian Mixture Model based on eviction data.
    
    This function processes the input dataframe to create a weighted representation
    of eviction data, fits a Bayesian Gaussian Mixture Model to the geographic coordinates,
    and assigns cluster labels to each original data point based on the most frequent
    cluster assignment among its weighted replications.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at minimum 'eviction_sum', 'longitude', and 'latitude' columns.
        
    Returns
    -------
    tuple
        A tuple containing:
        - bgmm : BayesianGaussianMixture
            The fitted Bayesian Gaussian Mixture Model.
        - cluster_map : pandas.Series
            A mapping from original dataframe indices to assigned cluster labels.
    """
    X_rep, replicated_df = feature_processing(df)
    bgmm = BayesianGaussianMixture(n_components=30, random_state=0)
    replicated_df['cluster'] = bgmm.fit_predict(X_rep)
    cluster_map = replicated_df.groupby(replicated_df.index)['cluster'].agg(lambda x: x.value_counts().idxmax())
    return bgmm, cluster_map

def logistic_model(X, Y):
    """
    Create and evaluate a logistic regression model for classification.
    
    This function builds a pipeline with standardization and logistic regression,
    performs cross-validation, trains the model on the training set, and evaluates
    it on the test set.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix containing predictor variables. Columns 'cluster' and 'NTA'
        will be dropped if present.
    Y : pandas.Series or array-like
        Target variable for classification.
        
    Returns
    -------
    tuple
        A tuple containing:
        - pipeline : sklearn.pipeline.Pipeline
            The fitted pipeline with StandardScaler and LogisticRegression.
        - cv_score_mean : float
            Mean cross-validation score from 5-fold cross-validation.
        - cm : numpy.ndarray
            Confusion matrix of predictions on the test set.
        - report : str
            Classification report including precision, recall, f1-score, and support.
    """
    X = X.drop(columns=[col for col in ['cluster', 'NTA'] if col in X.columns])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0, stratify=Y
    )
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    return pipeline, cv_scores.mean(), cm, classification_report(y_test, y_pred)
    
def RandomForest_model(X, Y):
    """
    Create and evaluate a Random Forest model for classification.
    
    This function builds a pipeline with standardization and Random Forest classifier,
    performs cross-validation, trains the model on the training set, and evaluates
    it on the test set.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix containing predictor variables. Columns 'cluster' and 'NTA'
        will be dropped if present.
    Y : pandas.Series or array-like
        Target variable for classification.
        
    Returns
    -------
    tuple
        A tuple containing:
        - pipeline : sklearn.pipeline.Pipeline
            The fitted pipeline with StandardScaler and RandomForestClassifier.
        - cv_score_mean : float
            Mean cross-validation score from 5-fold cross-validation.
        - cm : numpy.ndarray
            Confusion matrix of predictions on the test set.
        - report : str
            Classification report including precision, recall, f1-score, and support.
    """
    X = X.drop(columns=[col for col in ['cluster', 'NTA'] if col in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0, stratify=Y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Optional for trees, but harmless
        ('rf', RandomForestClassifier(
            n_estimators=100,
            random_state=0,
            class_weight='balanced',
            max_depth=None,
            n_jobs=-1
        ))
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    return pipeline, cv_scores.mean(), cm, classification_report(y_test, y_pred)