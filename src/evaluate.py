"""
Model Evaluation Module for Support Ticket Classification

This module provides comprehensive evaluation functions for classification models.
It includes metrics calculation, visualization, and performance analysis.

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def calculate_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray = None,
                                   average: str = 'macro') -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray, optional): Predicted probabilities
        average (str): Averaging method for multi-class metrics
        
    Returns:
        Dict[str, float]: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Weighted metrics
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Macro metrics (explicitly named for clarity)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC metrics (if probabilities are provided)
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
            else:  # Multi-class classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, average=average, multi_class='ovr')
                metrics['auc_pr'] = average_precision_score(y_true, y_prob, average=average)
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {str(e)}")
    
    return metrics


def print_classification_metrics(metrics: Dict[str, float], 
                              class_names: List[str] = None,
                              model_name: str = "Model") -> None:
    """
    Print classification metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Metrics dictionary
        class_names (List[str]): Names of the classes
        model_name (str): Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    
    print(f"\n📊 WEIGHTED METRICS:")
    print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    # AUC metrics if available
    if 'auc_roc' in metrics:
        print(f"\n📊 AUC METRICS:")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    
    # Per-class metrics
    if class_names is not None and 'precision_per_class' in metrics:
        print(f"\n📊 PER-CLASS METRICS:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*60}")
        
        for i, class_name in enumerate(class_names):
            if i < len(metrics['precision_per_class']):
                precision = metrics['precision_per_class'][i]
                recall = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                print(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: List[str],
                         model_name: str = "Model",
                         save_path: str = None,
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): Names of the classes
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add grid
    plt.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    # Only show plot if not in headless environment
    try:
        plt.show()
    except:
        plt.close()


def plot_classification_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                                        save_path: str = None,
                                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot comparison of metrics across multiple models.
    
    Args:
        metrics_dict (Dict[str, Dict[str, float]]): Dictionary of model metrics
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    # Extract metric names
    metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Prepare data for plotting
    models = list(metrics_dict.keys())
    data = []
    
    for model in models:
        for metric in metric_names:
            if metric in metrics_dict[model]:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics_dict[model][metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create grouped bar plot
    sns.barplot(data=df, x='Metric', y='Value', hue='Model', alpha=0.8)
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {save_path}")
    
    plt.show()


def plot_roc_curves(y_true: np.ndarray, 
                   y_prob_dict: Dict[str, np.ndarray],
                   class_names: List[str] = None,
                   save_path: str = None,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot ROC curves for multiple models.
    
    Args:
        y_true (np.ndarray): True labels
        y_prob_dict (Dict[str, np.ndarray]): Dictionary of model probabilities
        class_names (List[str]): Names of the classes
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)
    
    for model_name, y_prob in y_prob_dict.items():
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                auc = roc_auc_score(y_true, y_prob[:, 1])
                plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
            else:  # Multi-class classification
                # Plot ROC for each class (one-vs-rest)
                for i in range(len(class_names)):
                    fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                    auc = roc_auc_score(y_true == i, y_prob[:, i])
                    plt.plot(fpr, tpr, linewidth=1, alpha=0.7,
                            label=f'{model_name} - {class_names[i]} (AUC = {auc:.3f})')
        except Exception as e:
            print(f"Warning: Could not plot ROC curve for {model_name}: {str(e)}")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.title('ROC Curves', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved to: {save_path}")
    
    plt.show()


def evaluate_model(y_true: np.ndarray, 
                 y_pred: np.ndarray,
                 class_names: List[str],
                 model_name: str = "Model",
                 y_prob: np.ndarray = None,
                 save_plots: bool = True,
                 save_dir: str = "../plots") -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): Names of the classes
        model_name (str): Name of the model
        y_prob (np.ndarray, optional): Predicted probabilities
        save_plots (bool): Whether to save plots
        save_dir (str): Directory to save plots
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    # Print metrics
    print_classification_metrics(metrics, class_names, model_name)
    
    # Print detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Plot confusion matrix
    if save_plots:
        cm_save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, class_names, model_name, cm_save_path)
    else:
        plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    
    # Compile results
    results = {
        'metrics': metrics,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        'model_name': model_name
    }
    
    # Add probabilities if available
    if y_prob is not None:
        results['probabilities'] = y_prob
    
    return results


def compare_models(model_results: Dict[str, Dict[str, Any]],
                  save_plots: bool = True,
                  save_dir: str = "../plots") -> None:
    """
    Compare multiple models and create comparison plots.
    
    Args:
        model_results (Dict[str, Dict[str, Any]]): Dictionary of model results
        save_plots (bool): Whether to save plots
        save_dir (str): Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_dict = {}
    roc_data = {}
    
    for model_name, results in model_results.items():
        metrics_dict[model_name] = results['metrics']
        if 'probabilities' in results:
            roc_data[model_name] = results['probabilities']
    
    # Plot metrics comparison
    if save_plots:
        comparison_save_path = os.path.join(save_dir, "model_comparison.png")
        plot_classification_metrics_comparison(metrics_dict, comparison_save_path)
    else:
        plot_classification_metrics_comparison(metrics_dict)
    
    # Plot ROC curves if probabilities are available
    if roc_data:
        if save_plots:
            roc_save_path = os.path.join(save_dir, "roc_curves.png")
            plot_roc_curves(model_results[list(model_results.keys())[0]]['confusion_matrix'].shape[0], 
                          roc_data, save_path=roc_save_path)
        else:
            plot_roc_curves(model_results[list(model_results.keys())[0]]['confusion_matrix'].shape[0], 
                          roc_data)
    
    # Create summary table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Create DataFrame for easy comparison
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision (Macro)': metrics['precision_macro'],
            'Recall (Macro)': metrics['recall_macro'],
            'F1-Score (Macro)': metrics['f1_macro']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # Find best model for each metric
    best_models = {}
    for metric in ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']:
        best_model = df_comparison.loc[df_comparison[metric].idxmax(), 'Model']
        best_models[metric] = best_model
    
    print(f"\n🏆 BEST MODELS BY METRIC:")
    for metric, model in best_models.items():
        print(f"  {metric}: {model}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)
    
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)
    
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    # Evaluate models
    rf_results = evaluate_model(y_test, rf_pred, class_names, "Random Forest", rf_prob)
    lr_results = evaluate_model(y_test, lr_pred, class_names, "Logistic Regression", lr_prob)
    
    # Compare models
    compare_models({
        'Random Forest': rf_results,
        'Logistic Regression': lr_results
    })
