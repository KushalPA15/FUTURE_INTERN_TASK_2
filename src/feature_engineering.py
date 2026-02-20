"""
Feature Engineering Module for Support Ticket Classification

This module handles text vectorization and feature extraction for support tickets.
It provides functions to convert text data into numerical features suitable for ML models.

Author: ML Engineering Team
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def vectorize_text(train_text: pd.Series, 
                  test_text: pd.Series = None,
                  max_features: int = 5000,
                  ngram_range: Tuple[int, int] = (1, 2),
                  min_df: int = 2,
                  max_df: float = 0.95) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Convert text data to TF-IDF features.
    
    This function creates TF-IDF vectors from text data, which captures
    the importance of words in documents relative to the entire corpus.
    
    Args:
        train_text (pd.Series): Training text data
        test_text (pd.Series, optional): Test text data. If None, splits train_text
        max_features (int): Maximum number of features to keep
        ngram_range (Tuple[int, int]): Range of n-grams to include
        min_df (int): Minimum document frequency for a term
        max_df (float): Maximum document frequency for a term
        
    Returns:
        Tuple[np.ndarray, np.ndarray, TfidfVectorizer]: 
            - X_train: TF-IDF features for training data
            - X_test: TF-IDF features for test data  
            - vectorizer: Fitted TF-IDF vectorizer
            
    Example:
        >>> X_train, X_test, vectorizer = vectorize_text(train_texts, test_texts)
    """
    print("Creating TF-IDF features...")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',  # Additional stopword filtering
        lowercase=True,       # Ensure lowercase
        strip_accents='unicode'  # Remove accents
    )
    
    # Fit and transform training data
    X_train = vectorizer.fit_transform(train_text)
    
    # Transform test data if provided, otherwise split training data
    if test_text is not None:
        X_test = vectorizer.transform(test_text)
    else:
        # Split training data for validation
        X_train, X_test, _, _ = train_test_split(
            X_train, train_text, test_size=0.2, random_state=42
        )
    
    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")
    
    # Display feature information
    feature_names = vectorizer.get_feature_names_out()
    print(f"Total features created: {len(feature_names)}")
    
    # Show some example features
    print(f"Sample features: {list(feature_names[:10])}")
    
    return X_train, X_test, vectorizer


def encode_labels(y_train: pd.Series, y_test: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Dict]:
    """
    Encode categorical labels to numerical format.
    
    Args:
        y_train (pd.Series): Training labels
        y_test (pd.Series, optional): Test labels
        
    Returns:
        Tuple[np.ndarray, np.ndarray, LabelEncoder, Dict]:
            - y_train_encoded: Encoded training labels
            - y_test_encoded: Encoded test labels
            - label_encoder: Fitted label encoder
            - label_mapping: Dictionary mapping original labels to encoded values
    """
    print("Encoding labels...")
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    # Fit and transform training labels
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Transform test labels if provided
    if y_test is not None:
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = None
    
    # Create label mapping
    label_mapping = {
        original: encoded for original, encoded in 
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    }
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Class distribution (training):")
    for class_name, class_encoded in label_mapping.items():
        count = np.sum(y_train_encoded == class_encoded)
        print(f"  {class_name}: {count} samples")
    
    return y_train_encoded, y_test_encoded, label_encoder, label_mapping


def prepare_data_splits(df: pd.DataFrame,
                       text_column: str = 'cleaned_text',
                       target_column: str = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare train/test splits for modeling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            - X_train: Training text data
            - X_test: Test text data
            - y_train: Training labels
            - y_test: Test labels
    """
    if target_column is None:
        raise ValueError("target_column must be specified")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")
    
    print(f"Preparing data splits with test_size={test_size}")
    
    # Extract features and target
    X = df[text_column]
    y = df[target_column]
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def create_feature_pipeline(df: pd.DataFrame,
                           text_column: str = 'cleaned_text',
                           target_column: str = None,
                           max_features: int = 5000,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Complete feature engineering pipeline.
    
    This function handles the entire feature engineering process:
    1. Data splitting
    2. Text vectorization
    3. Label encoding
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        target_column (str): Name of the target column
        max_features (int): Maximum number of TF-IDF features
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Dict[str, Any]: Dictionary containing all processed data and objects
    """
    if target_column is None:
        raise ValueError("target_column must be specified")
    
    print(f"Starting feature engineering for target: {target_column}")
    print("=" * 50)
    
    # 1. Prepare data splits
    X_train, X_test, y_train, y_test = prepare_data_splits(
        df, text_column, target_column, test_size, random_state
    )
    
    # 2. Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train, X_test, max_features=max_features
    )
    
    # 3. Encode labels
    y_train_enc, y_test_enc, label_encoder, label_mapping = encode_labels(
        y_train, y_test
    )
    
    # Package results
    pipeline_results = {
        'X_train': X_train_vec,
        'X_test': X_test_vec,
        'y_train': y_train_enc,
        'y_test': y_test_enc,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'label_mapping': label_mapping,
        'feature_names': vectorizer.get_feature_names_out(),
        'n_features': len(vectorizer.get_feature_names_out()),
        'n_classes': len(label_encoder.classes_)
    }
    
    print("=" * 50)
    print(f"Feature engineering completed for {target_column}")
    print(f"Features: {pipeline_results['n_features']}")
    print(f"Classes: {pipeline_results['n_classes']}")
    
    return pipeline_results


def save_feature_artifacts(vectorizer: LabelEncoder, 
                          label_encoder: LabelEncoder,
                          save_dir: str = "../models") -> None:
    """
    Save fitted vectorizer and label encoder for later use.
    
    Args:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        label_encoder (LabelEncoder): Fitted label encoder
        save_dir (str): Directory to save the artifacts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save vectorizer
    vectorizer_path = os.path.join(save_dir, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save label encoder
    label_encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to: {label_encoder_path}")


def get_top_features(vectorizer: TfidfVectorizer, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    label_encoder: LabelEncoder,
                    n_top: int = 10) -> Dict[str, list]:
    """
    Get top features for each class based on mean TF-IDF scores.
    
    Args:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        label_encoder (LabelEncoder): Fitted label encoder
        n_top (int): Number of top features to return per class
        
    Returns:
        Dict[str, list]: Dictionary mapping class names to top features
    """
    feature_names = vectorizer.get_feature_names_out()
    class_names = label_encoder.classes_
    
    top_features = {}
    
    for class_idx, class_name in enumerate(class_names):
        # Get documents for this class
        class_mask = (y == class_idx)
        class_X = X[class_mask]
        
        # Calculate mean TF-IDF score for each feature
        mean_scores = np.mean(class_X.toarray(), axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_scores)[-n_top:][::-1]
        top_features[class_name] = [
            (feature_names[i], mean_scores[i]) for i in top_indices
        ]
    
    return top_features


if __name__ == "__main__":
    # Example usage
    try:
        # Load preprocessed data
        df = pd.read_csv("../data/tickets.csv")
        
        # Example for category classification
        category_pipeline = create_feature_pipeline(
            df, 
            text_column='Ticket Description',
            target_column='Ticket Type'
        )
        
        # Get top features for each category
        top_features = get_top_features(
            category_pipeline['vectorizer'],
            category_pipeline['X_train'],
            category_pipeline['y_train'],
            category_pipeline['label_encoder']
        )
        
        print("\nTop features by category:")
        for category, features in top_features.items():
            print(f"\n{category}:")
            for feature, score in features[:5]:
                print(f"  {feature}: {score:.4f}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
