"""
Utilities Module for Support Ticket Classification

This module contains utility functions for data loading, model management,
and common operations used across the project.

Author: ML Engineering Team
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("support_ticket_classifier")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_model_metadata(metadata: Dict[str, Any], 
                       save_path: str) -> None:
    """
    Save model metadata to JSON file.
    
    Args:
        metadata (Dict[str, Any]): Metadata dictionary
        save_path (str): Path to save the metadata
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Convert and save
    converted_metadata = convert_numpy(metadata)
    
    with open(save_path, 'w') as f:
        json.dump(converted_metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to: {save_path}")


def load_model_metadata(load_path: str) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        load_path (str): Path to the metadata file
        
    Returns:
        Dict[str, Any]: Loaded metadata
    """
    try:
        with open(load_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {load_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {load_path}: {str(e)}")


def create_experiment_directory(base_dir: str = "experiments") -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        str: Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["models", "plots", "logs", "results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def validate_data_quality(df: pd.DataFrame, 
                         text_column: str = 'cleaned_text',
                         target_column: str = None) -> Dict[str, Any]:
    """
    Validate data quality and return statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        target_column (str): Name of the target column
        
    Returns:
        Dict[str, Any]: Data quality statistics
    """
    quality_report = {}
    
    # Basic statistics
    quality_report['total_rows'] = len(df)
    quality_report['total_columns'] = len(df.columns)
    quality_report['missing_values'] = df.isnull().sum().to_dict()
    
    # Text column statistics
    if text_column in df.columns:
        text_lengths = df[text_column].str.len()
        quality_report['text_stats'] = {
            'mean_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'empty_texts': (text_lengths == 0).sum(),
            'very_short_texts': (text_lengths < 10).sum()
        }
    
    # Target column statistics
    if target_column and target_column in df.columns:
        target_counts = df[target_column].value_counts()
        quality_report['target_stats'] = {
            'unique_classes': len(target_counts),
            'class_distribution': target_counts.to_dict(),
            'class_balance_ratio': target_counts.max() / target_counts.min(),
            'majority_class': target_counts.index[0],
            'minority_class': target_counts.index[-1]
        }
    
    # Duplicate rows
    quality_report['duplicate_rows'] = df.duplicated().sum()
    
    return quality_report


def plot_data_distribution(df: pd.DataFrame,
                          columns: List[str],
                          save_path: str = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot distribution of specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (List[str]): List of columns to plot
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # Max 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, column in enumerate(columns):
        row = i // 3
        col = i % 3
        
        if column in df.columns:
            if df[column].dtype == 'object':
                # Categorical data
                df[column].value_counts().plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'Distribution of {column}')
                axes[row, col].tick_params(axis='x', rotation=45)
            else:
                # Numerical data
                df[column].hist(bins=30, ax=axes[row, col])
                axes[row, col].set_title(f'Distribution of {column}')
        
        axes[row, col].set_xlabel(column)
        axes[row, col].set_ylabel('Count')
    
    # Hide empty subplots
    for i in range(n_cols, n_rows * 3):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()


def create_model_card(model_name: str,
                    model_type: str,
                    performance_metrics: Dict[str, float],
                    training_data_info: Dict[str, Any],
                    model_parameters: Dict[str, Any],
                    feature_importance: Dict[str, List] = None,
                    limitations: List[str] = None,
                    intended_use: str = None) -> Dict[str, Any]:
    """
    Create a comprehensive model card for documentation.
    
    Args:
        model_name (str): Name of the model
        model_type (str): Type of model (classification, regression, etc.)
        performance_metrics (Dict[str, float]): Performance metrics
        training_data_info (Dict[str, Any]): Information about training data
        model_parameters (Dict[str, Any]): Model hyperparameters
        feature_importance (Dict[str, List]): Important features per class
        limitations (List[str]): Model limitations
        intended_use (str): Intended use case
        
    Returns:
        Dict[str, Any]: Model card dictionary
    """
    model_card = {
        'model_name': model_name,
        'model_type': model_type,
        'created_date': datetime.now().isoformat(),
        'version': '1.0',
        
        'performance_metrics': performance_metrics,
        'model_parameters': model_parameters,
        'training_data_info': training_data_info,
        
        'feature_importance': feature_importance or {},
        
        'limitations': limitations or [
            'Model performance may degrade with significantly different text patterns',
            'Model is trained on English language tickets only',
            'Model may not handle domain-specific terminology well'
        ],
        
        'intended_use': intended_use or f'Classification of {model_name.lower()} for support tickets',
        
        'ethical_considerations': [
            'Model should be regularly monitored for bias',
            'Human oversight is recommended for critical decisions',
            'Model predictions should be validated before deployment'
        ]
    }
    
    return model_card


def save_model_card(model_card: Dict[str, Any], save_path: str) -> None:
    """
    Save model card to JSON file.
    
    Args:
        model_card (Dict[str, Any]): Model card dictionary
        save_path (str): Path to save the model card
    """
    save_model_metadata(model_card, save_path)
    print(f"Model card saved to: {save_path}")


def load_and_validate_model(model_path: str, 
                          metadata_path: str = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and validate with metadata.
    
    Args:
        model_path (str): Path to the saved model
        metadata_path (str, optional): Path to the metadata file
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Loaded model and metadata
    """
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {str(e)}")
    
    # Load metadata if provided
    metadata = {}
    if metadata_path:
        try:
            metadata = load_model_metadata(metadata_path)
            print(f"Metadata loaded successfully from: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {str(e)}")
    
    return model, metadata


def batch_predict(texts: List[str],
                 model_path: str,
                 vectorizer_path: str,
                 label_encoder_path: str,
                 batch_size: int = 100) -> Tuple[List[str], List[float]]:
    """
    Make batch predictions on a list of texts.
    
    Args:
        texts (List[str]): List of input texts
        model_path (str): Path to the saved model
        vectorizer_path (str): Path to the saved vectorizer
        label_encoder_path (str): Path to the saved label encoder
        batch_size (int): Batch size for processing
        
    Returns:
        Tuple[List[str], List[float]]: Predictions and confidence scores
    """
    # Load artifacts
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Load preprocessing function
    from data_preprocessing import clean_text
    
    predictions = []
    confidences = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Clean texts
        cleaned_texts = [clean_text(text) for text in batch_texts]
        
        # Vectorize
        text_vectors = vectorizer.transform(cleaned_texts)
        
        # Predict
        batch_predictions = model.predict(text_vectors)
        batch_labels = label_encoder.inverse_transform(batch_predictions)
        
        # Get confidences
        if hasattr(model, 'predict_proba'):
            batch_confidences = np.max(model.predict_proba(text_vectors), axis=1)
        else:
            batch_confidences = [None] * len(batch_predictions)
        
        predictions.extend(batch_labels)
        confidences.extend(batch_confidences)
    
    return predictions, confidences


def create_prediction_report(texts: List[str],
                           predictions: List[str],
                           confidences: List[float] = None,
                           save_path: str = None) -> pd.DataFrame:
    """
    Create a prediction report.
    
    Args:
        texts (List[str]): Input texts
        predictions (List[str]): Model predictions
        confidences (List[float]): Confidence scores
        save_path (str, optional): Path to save the report
        
    Returns:
        pd.DataFrame: Prediction report
    """
    report_data = {
        'text': texts,
        'prediction': predictions
    }
    
    if confidences:
        report_data['confidence'] = confidences
    
    df = pd.DataFrame(report_data)
    
    # Add summary statistics
    if confidences:
        summary_stats = {
            'total_predictions': len(df),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'prediction_distribution': df['prediction'].value_counts().to_dict()
        }
    else:
        summary_stats = {
            'total_predictions': len(df),
            'prediction_distribution': df['prediction'].value_counts().to_dict()
        }
    
    print("Prediction Summary:")
    for key, value in summary_stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Prediction report saved to: {save_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    logger = setup_logging("INFO", "logs/utils.log")
    logger.info("Utilities module loaded successfully")
    
    # Example experiment directory creation
    exp_dir = create_experiment_directory()
    print(f"Experiment directory: {exp_dir}")
    
    # Example model card creation
    model_card = create_model_card(
        model_name="Category Classifier",
        model_type="Multiclass Classification",
        performance_metrics={"accuracy": 0.85, "f1_macro": 0.83},
        training_data_info={"samples": 8000, "features": 5000},
        model_parameters={"C": 1.0, "penalty": "l2"}
    )
    
    save_model_card(model_card, f"{exp_dir}/results/model_card.json")
