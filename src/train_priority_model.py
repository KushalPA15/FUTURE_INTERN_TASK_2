"""
Priority Prediction Model Training Module

This module trains machine learning models to predict support ticket priority levels.
It includes model training, comparison, and saving functionality.

Author: ML Engineering Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preprocessing import load_and_preprocess_data
from feature_engineering import create_feature_pipeline, get_top_features, save_feature_artifacts
from evaluate import evaluate_model, plot_confusion_matrix, print_classification_metrics


def train_multiple_models_with_class_weights(X_train: np.ndarray, 
                                           y_train: np.ndarray,
                                           models_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Train multiple classification models with class weight handling for imbalanced data.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        models_config (Dict[str, Any]): Configuration for models
        
    Returns:
        Dict[str, Any]: Dictionary containing trained models and their performance
    """
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    if models_config is None:
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                }
            },
            'Multinomial Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_prior': [True, False]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Linear SVM': {
                'model': LinearSVC(random_state=42, max_iter=2000, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1.0, 10.0]
                }
            }
        }
    
    trained_models = {}
    model_scores = {}
    
    print("Training multiple models with class weight handling...")
    print("=" * 60)
    print(f"Class weights: {class_weight_dict}")
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            scoring='f1_macro',  # Use F1-macro for imbalanced classes
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Perform cross-validation with F1-macro scoring
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_macro')
        
        # Store results
        trained_models[model_name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'grid_search_score': grid_search.best_score_
        }
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Cross-validation F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Grid search F1-macro: {grid_search.best_score_:.4f}")
    
    # Find best model based on cross-validation F1-macro score
    best_model_name = max(trained_models.keys(), 
                         key=lambda x: trained_models[x]['cv_score_mean'])
    
    print(f"\n" + "=" * 60)
    print(f"Best performing model: {best_model_name}")
    print(f"Best CV F1-macro: {trained_models[best_model_name]['cv_score_mean']:.4f}")
    
    return {
        'models': trained_models,
        'best_model_name': best_model_name,
        'model_scores': model_scores,
        'class_weights': class_weight_dict
    }


def train_priority_model(data_path: str = "data/tickets.csv",
                        save_models: bool = True,
                        save_dir: str = "models") -> Dict[str, Any]:
    """
    Complete pipeline for training priority prediction model.
    
    Args:
        data_path (str): Path to the dataset
        save_models (bool): Whether to save trained models
        save_dir (str): Directory to save models
        
    Returns:
        Dict[str, Any]: Dictionary containing training results and models
    """
    print("Starting Priority Prediction Model Training")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    # Analyze priority distribution
    print(f"\nPriority Distribution Analysis:")
    priority_counts = df['Ticket Priority'].value_counts()
    priority_percentages = df['Ticket Priority'].value_counts(normalize=True) * 100
    
    for priority in priority_counts.index:
        count = priority_counts[priority]
        percentage = priority_percentages[priority]
        print(f"  {priority}: {count} tickets ({percentage:.1f}%)")
    
    # 2. Feature engineering
    print("\nStep 2: Feature engineering...")
    feature_pipeline = create_feature_pipeline(
        df,
        text_column='cleaned_text',
        target_column='Ticket Priority',
        max_features=5000,
        test_size=0.2,
        random_state=42
    )
    
    # 3. Train multiple models with class weight handling
    print("\nStep 3: Training models with class balancing...")
    training_results = train_multiple_models_with_class_weights(
        feature_pipeline['X_train'],
        feature_pipeline['y_train']
    )
    
    # 4. Evaluate best model on test set
    print("\nStep 4: Evaluating best model...")
    best_model_name = training_results['best_model_name']
    best_model = training_results['models'][best_model_name]['model']
    
    # Make predictions on test set
    y_pred = best_model.predict(feature_pipeline['X_test'])
    
    # Evaluate performance
    test_results = evaluate_model(
        feature_pipeline['y_test'],
        y_pred,
        feature_pipeline['label_encoder'].classes_,
        model_name=best_model_name
    )
    
    # 5. Get top features for each priority level
    print("\nStep 5: Extracting important features...")
    top_features = get_top_features(
        feature_pipeline['vectorizer'],
        feature_pipeline['X_train'],
        feature_pipeline['y_train'],
        feature_pipeline['label_encoder'],
        n_top=10
    )
    
    # Display top features
    print("\nTop features by priority level:")
    for priority, features in top_features.items():
        print(f"\n{priority} Priority:")
        for i, (feature, score) in enumerate(features[:5], 1):
            print(f"  {i}. {feature}: {score:.4f}")
    
    # 6. Save models and artifacts
    if save_models:
        print("\nStep 6: Saving models and artifacts...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the best model
        model_path = os.path.join(save_dir, "priority_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best model saved to: {model_path}")
        
        # Save priority-specific vectorizer and label encoder
        priority_vectorizer_path = os.path.join(save_dir, "priority_vectorizer.pkl")
        priority_label_encoder_path = os.path.join(save_dir, "priority_label_encoder.pkl")
        
        joblib.dump(feature_pipeline['vectorizer'], priority_vectorizer_path)
        joblib.dump(feature_pipeline['label_encoder'], priority_label_encoder_path)
        
        print(f"Priority vectorizer saved to: {priority_vectorizer_path}")
        print(f"Priority label encoder saved to: {priority_label_encoder_path}")
        
        # Save training results
        results_path = os.path.join(save_dir, "priority_training_results.pkl")
        joblib.dump({
            'model_name': best_model_name,
            'model_params': training_results['models'][best_model_name]['best_params'],
            'cv_score': training_results['models'][best_model_name]['cv_score_mean'],
            'test_results': test_results,
            'top_features': top_features,
            'label_mapping': feature_pipeline['label_mapping'],
            'class_weights': training_results['class_weights']
        }, results_path)
        print(f"Training results saved to: {results_path}")
    
    # 7. Compile final results
    final_results = {
        'model_name': best_model_name,
        'model': best_model,
        'model_params': training_results['models'][best_model_name]['best_params'],
        'cv_score': training_results['models'][best_model_name]['cv_score_mean'],
        'cv_score_std': training_results['models'][best_model_name]['cv_score_std'],
        'test_accuracy': test_results['accuracy'],
        'test_f1_macro': test_results['f1_macro'],
        'test_results': test_results,
        'top_features': top_features,
        'label_mapping': feature_pipeline['label_mapping'],
        'class_weights': training_results['class_weights'],
        'feature_pipeline': feature_pipeline,
        'training_results': training_results
    }
    
    print("\n" + "=" * 60)
    print("Priority Prediction Training Complete!")
    print(f"Best Model: {best_model_name}")
    print(f"Cross-Validation F1-Macro: {final_results['cv_score']:.4f}")
    print(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Test F1-Macro: {final_results['test_f1_macro']:.4f}")
    
    return final_results


def predict_priority(text: str, 
                    model_path: str = "../models/priority_model.pkl",
                    vectorizer_path: str = "../models/priority_vectorizer.pkl",
                    label_encoder_path: str = "../models/priority_label_encoder.pkl") -> Tuple[str, float]:
    """
    Make predictions on new text using the trained priority model.
    
    Args:
        text (str): Input text to classify
        model_path (str): Path to saved model
        vectorizer_path (str): Path to saved vectorizer
        label_encoder_path (str): Path to saved label encoder
        
    Returns:
        Tuple[str, float]: Predicted priority and confidence score
    """
    try:
        # Load artifacts
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(label_encoder_path)
        
        # Load preprocessing function
        from data_preprocessing import clean_text
        
        # Clean and vectorize text
        cleaned_text = clean_text(text)
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)
        predicted_priority = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
        else:
            # For models without predict_proba, use decision function if available
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(text_vector)[0]
                if len(decision_scores.shape) == 0:  # Binary classification
                    confidence = abs(decision_scores)
                else:  # Multi-class
                    confidence = max(decision_scores)
            else:
                confidence = None
        
        return predicted_priority, confidence
        
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")


def analyze_priority_prediction_patterns(results: Dict[str, Any]) -> None:
    """
    Analyze patterns in priority predictions to provide business insights.
    
    Args:
        results (Dict[str, Any]): Training results dictionary
    """
    print("\n" + "=" * 60)
    print("PRIORITY PATTERN ANALYSIS")
    print("=" * 60)
    
    top_features = results['top_features']
    label_mapping = results['label_mapping']
    
    # Analyze critical priority indicators
    if 'Critical' in top_features:
        print("\n🚨 CRITICAL PRIORITY INDICATORS:")
        critical_features = top_features['Critical'][:5]
        for i, (feature, score) in enumerate(critical_features, 1):
            print(f"  {i}. '{feature}' (score: {score:.4f})")
    
    # Analyze high priority indicators
    if 'High' in top_features:
        print("\n⚡ HIGH PRIORITY INDICATORS:")
        high_features = top_features['High'][:5]
        for i, (feature, score) in enumerate(high_features, 1):
            print(f"  {i}. '{feature}' (score: {score:.4f})")
    
    # Analyze low priority indicators
    if 'Low' in top_features:
        print("\n📝 LOW PRIORITY INDICATORS:")
        low_features = top_features['Low'][:5]
        for i, (feature, score) in enumerate(low_features, 1):
            print(f"  {i}. '{feature}' (score: {score:.4f})")
    
    # Business recommendations
    print("\n💡 BUSINESS RECOMMENDATIONS:")
    print("  1. Set up automated alerts for tickets containing critical indicators")
    print("  2. Route high-priority tickets to senior support agents")
    print("  3. Use low-priority indicators for automated responses or self-service")
    print("  4. Monitor trending keywords to proactively identify potential issues")


if __name__ == "__main__":
    # Train the priority prediction model
    try:
        results = train_priority_model()
        
        # Analyze prediction patterns
        analyze_priority_prediction_patterns(results)
        
        # Example predictions
        print("\n" + "=" * 60)
        print("Example Predictions:")
        
        example_texts = [
            "URGENT: System completely down, no access to critical data",
            "I have a question about my billing statement",
            "The application is running slowly but I can still work",
            "Please help me reset my password, it's not urgent"
        ]
        
        for text in example_texts:
            try:
                priority, confidence = predict_priority(text)
                print(f"Text: '{text}'")
                print(f"Predicted Priority: {priority}")
                if confidence:
                    print(f"Confidence: {confidence:.4f}")
                print("-" * 40)
            except Exception as e:
                print(f"Error predicting '{text}': {str(e)}")
                
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
