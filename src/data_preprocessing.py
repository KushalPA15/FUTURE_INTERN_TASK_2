"""
Data Preprocessing Module for Support Ticket Classification

This module handles text cleaning and preprocessing operations for support tickets.
It provides functions to clean raw ticket text data for machine learning models.

Author: ML Engineering Team
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def clean_text(text: str, 
               remove_punctuation: bool = True,
               remove_stopwords: bool = True,
               lemmatize: bool = True,
               lowercase: bool = True) -> str:
    """
    Clean and preprocess text data for NLP tasks.
    
    This function performs comprehensive text cleaning including:
    - Lowercase conversion
    - Punctuation removal
    - Stopword removal
    - Tokenization
    - Lemmatization
    
    Args:
        text (str): Raw text input to be cleaned
        remove_punctuation (bool): Whether to remove punctuation marks
        remove_stopwords (bool): Whether to remove common stopwords
        lemmatize (bool): Whether to apply lemmatization
        lowercase (bool): Whether to convert text to lowercase
        
    Returns:
        str: Cleaned and preprocessed text
        
    Example:
        >>> clean_text("I'm having issues with my product!")
        'having issues product'
    """
    if not isinstance(text, str):
        return ""
    
    if pd.isna(text) or text.strip() == "":
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b', '', text)
    
    # Remove numbers (optional - keeping for now as they might be important)
    # text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Add custom stopwords relevant to support tickets
        custom_stopwords = {'please', 'thank', 'thanks', 'hi', 'hello', 'dear', 'regards', 'sincerely'}
        stop_words.update(custom_stopwords)
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove empty tokens and single characters
    tokens = [token for token in tokens if len(token) > 1 and token.strip()]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to text
    cleaned_text = ' '.join(tokens)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str = 'Ticket Description',
                        cleaned_column: str = 'cleaned_text') -> pd.DataFrame:
    """
    Apply text cleaning to a pandas DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing text data
        text_column (str): Name of the column containing raw text
        cleaned_column (str): Name of the column to store cleaned text
        
    Returns:
        pd.DataFrame: DataFrame with added cleaned text column
        
    Raises:
        ValueError: If the specified text column doesn't exist
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Apply text cleaning
    print(f"Cleaning text data in column '{text_column}'...")
    df_processed[cleaned_column] = df_processed[text_column].apply(clean_text)
    
    # Remove rows where cleaned text is empty
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed[cleaned_column].str.len() > 0]
    final_count = len(df_processed)
    
    print(f"Removed {initial_count - final_count} rows with empty text")
    print(f"Processed {final_count} tickets successfully")
    
    return df_processed


def load_and_preprocess_data(file_path: str, 
                           text_column: str = 'Ticket Description',
                           category_column: str = 'Ticket Type',
                           priority_column: str = 'Ticket Priority') -> pd.DataFrame:
    """
    Load CSV data and apply preprocessing.
    
    Args:
        file_path (str): Path to the CSV file
        text_column (str): Name of the text column
        category_column (str): Name of the category column
        priority_column (str): Name of the priority column
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        # Load data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Check required columns
        required_columns = [text_column, category_column, priority_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values in target columns
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing values")
        
        # Apply text preprocessing
        df_processed = preprocess_dataframe(df, text_column)
        
        # Display basic statistics
        print(f"\nDataset Statistics:")
        print(f"Total tickets: {len(df_processed)}")
        print(f"Categories: {df_processed[category_column].nunique()}")
        print(f"Priorities: {df_processed[priority_column].nunique()}")
        
        print(f"\nCategory Distribution:")
        print(df_processed[category_column].value_counts())
        
        print(f"\nPriority Distribution:")
        print(df_processed[priority_column].value_counts())
        
        return df_processed
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading and preprocessing data: {str(e)}")


def get_text_statistics(df: pd.DataFrame, text_column: str = 'cleaned_text') -> dict:
    """
    Calculate basic statistics about the text data.
    
    Args:
        df (pd.DataFrame): DataFrame with cleaned text
        text_column (str): Name of the cleaned text column
        
    Returns:
        dict: Dictionary containing text statistics
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Calculate text lengths
    text_lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'total_documents': len(df),
        'avg_text_length': text_lengths.mean(),
        'median_text_length': text_lengths.median(),
        'min_text_length': text_lengths.min(),
        'max_text_length': text_lengths.max(),
        'avg_word_count': word_counts.mean(),
        'median_word_count': word_counts.median(),
        'min_word_count': word_counts.min(),
        'max_word_count': word_counts.max(),
        'empty_documents': (text_lengths == 0).sum()
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    try:
        # Load and preprocess data
        df = load_and_preprocess_data("../data/tickets.csv")
        
        # Get text statistics
        stats = get_text_statistics(df)
        print(f"\nText Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
