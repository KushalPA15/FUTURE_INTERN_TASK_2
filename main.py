"""
Main Entry Point for Support Ticket Classification System

This script provides a unified interface to run the complete ML pipeline
for support ticket classification and priority prediction.

Usage:
    python main.py --train-category    # Train category classification model
    python main.py --train-priority     # Train priority prediction model
    python main.py --train-all          # Train both models
    python main.py --predict            # Interactive prediction mode
    python main.py --evaluate           # Evaluate trained models

Author: ML Engineering Team
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from train_category_model import train_category_model, predict_category
from train_priority_model import train_priority_model, predict_priority
from evaluate import compare_models
from utils import setup_logging


def train_models(category: bool = True, priority: bool = True):
    """
    Train the specified models.
    
    Args:
        category (bool): Whether to train category model
        priority (bool): Whether to train priority model
    """
    logger = setup_logging("INFO", "logs/main.log")
    
    if category:
        logger.info("Starting Category Classification Model Training")
        print("\n" + "="*60)
        print("🚀 TRAINING CATEGORY CLASSIFICATION MODEL")
        print("="*60)
        
        try:
            category_results = train_category_model(
                data_path="data/tickets.csv",
                save_models=True,
                save_dir="models"
            )
            print("✅ Category model training completed successfully!")
        except Exception as e:
            print(f"❌ Error training category model: {str(e)}")
            logger.error(f"Category model training failed: {str(e)}")
    
    if priority:
        logger.info("Starting Priority Prediction Model Training")
        print("\n" + "="*60)
        print("🚀 TRAINING PRIORITY PREDICTION MODEL")
        print("="*60)
        
        try:
            priority_results = train_priority_model(
                data_path="data/tickets.csv",
                save_models=True,
                save_dir="models"
            )
            print("✅ Priority model training completed successfully!")
        except Exception as e:
            print(f"❌ Error training priority model: {str(e)}")
            logger.error(f"Priority model training failed: {str(e)}")


def interactive_predict():
    """
    Interactive prediction mode for testing models.
    """
    print("\n" + "="*60)
    print("🔮 INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter your support ticket text (or 'quit' to exit):")
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text.")
            continue
        
        print(f"\n📝 Analyzing: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Predict category
        try:
            category, cat_confidence = predict_category(text)
            print(f"🏷️  Category: {category}")
            if cat_confidence:
                print(f"📊 Confidence: {cat_confidence:.4f}")
        except Exception as e:
            print(f"❌ Category prediction error: {str(e)}")
        
        # Predict priority
        try:
            priority, priority_confidence = predict_priority(text)
            print(f"🚨 Priority: {priority}")
            if priority_confidence:
                print(f"📊 Confidence: {priority_confidence:.4f}")
        except Exception as e:
            print(f"❌ Priority prediction error: {str(e)}")
        
        print("-" * 40)


def evaluate_models():
    """
    Evaluate trained models and compare performance.
    """
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # This would require loading test results and comparing models
    # For now, we'll provide a placeholder
    print("📋 Model evaluation functionality would be implemented here.")
    print("   - Load trained models")
    print("   - Load test data")
    print("   - Generate predictions")
    print("   - Calculate metrics")
    print("   - Create comparison plots")
    
    # Check if models exist
    model_files = [
        "models/category_model.pkl",
        "models/priority_model.pkl",
        "models/vectorizer.pkl"
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print(f"\n⚠️  Missing model files: {missing_models}")
        print("Please train the models first using: python main.py --train-all")
    else:
        print("\n✅ All model files found. Evaluation ready.")


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="Support Ticket Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train-category     Train only category model
  python main.py --train-priority      Train only priority model
  python main.py --train-all           Train both models
  python main.py --predict              Interactive prediction
  python main.py --evaluate             Evaluate models
        """
    )
    
    parser.add_argument(
        "--train-category",
        action="store_true",
        help="Train the category classification model"
    )
    
    parser.add_argument(
        "--train-priority", 
        action="store_true",
        help="Train the priority prediction model"
    )
    
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train both category and priority models"
    )
    
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run interactive prediction mode"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate trained models"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Execute requested actions
    if args.train_all:
        train_models(category=True, priority=True)
    elif args.train_category:
        train_models(category=True, priority=False)
    elif args.train_priority:
        train_models(category=False, priority=True)
    elif args.predict:
        interactive_predict()
    elif args.evaluate:
        evaluate_models()


if __name__ == "__main__":
    main()
