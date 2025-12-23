#!/bin/bash

# X-ray Analysis System - Training Script
# This script provides easy commands for training different model architectures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
print_status "Checking dependencies..."
python -c "import torch, torchvision, numpy, pandas, sklearn, matplotlib, seaborn" 2>/dev/null || {
    print_error "Required packages not found. Please install requirements.txt"
    exit 1
}

# Create necessary directories
print_status "Creating directories..."
mkdir -p checkpoints logs data/raw data/processed assets/evaluation assets/explainability

# Function to train a model
train_model() {
    local config_file=$1
    local model_name=$2
    
    print_status "Training $model_name with config: $config_file"
    
    if [ ! -f "$config_file" ]; then
        print_error "Config file $config_file not found"
        exit 1
    fi
    
    python src/train/trainer.py "$config_file"
    
    if [ $? -eq 0 ]; then
        print_status "$model_name training completed successfully"
    else
        print_error "$model_name training failed"
        exit 1
    fi
}

# Function to evaluate a model
evaluate_model() {
    local checkpoint_path=$1
    local model_name=$2
    
    print_status "Evaluating $model_name with checkpoint: $checkpoint_path"
    
    if [ ! -f "$checkpoint_path" ]; then
        print_error "Checkpoint file $checkpoint_path not found"
        exit 1
    fi
    
    python src/eval/evaluator.py --checkpoint "$checkpoint_path" --explanations --num_explanations 20
    
    if [ $? -eq 0 ]; then
        print_status "$model_name evaluation completed successfully"
    else
        print_error "$model_name evaluation failed"
        exit 1
    fi
}

# Main script logic
case "$1" in
    "train")
        case "$2" in
            "resnet18")
                train_model "configs/default.yaml" "ResNet18"
                ;;
            "efficientnet")
                train_model "configs/efficientnet.yaml" "EfficientNet-B0"
                ;;
            "vit")
                train_model "configs/vit.yaml" "Vision Transformer"
                ;;
            "all")
                print_status "Training all models..."
                train_model "configs/default.yaml" "ResNet18"
                train_model "configs/efficientnet.yaml" "EfficientNet-B0"
                train_model "configs/vit.yaml" "Vision Transformer"
                ;;
            *)
                print_error "Unknown model: $2"
                echo "Usage: $0 train [resnet18|efficientnet|vit|all]"
                exit 1
                ;;
        esac
        ;;
    "eval")
        case "$2" in
            "resnet18")
                evaluate_model "checkpoints/checkpoint_best.pth" "ResNet18"
                ;;
            "efficientnet")
                evaluate_model "checkpoints/checkpoint_best.pth" "EfficientNet-B0"
                ;;
            "vit")
                evaluate_model "checkpoints/checkpoint_best.pth" "Vision Transformer"
                ;;
            *)
                print_error "Unknown model: $2"
                echo "Usage: $0 eval [resnet18|efficientnet|vit]"
                exit 1
                ;;
        esac
        ;;
    "demo")
        print_status "Launching Streamlit demo..."
        streamlit run demo/streamlit_app.py
        ;;
    "setup")
        print_status "Setting up the project..."
        
        # Install requirements
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        
        # Create directories
        print_status "Creating project directories..."
        mkdir -p checkpoints logs data/{raw,processed,external} assets/{evaluation,explainability,sample_images} tests
        
        # Set up pre-commit hooks
        if command -v pre-commit &> /dev/null; then
            print_status "Setting up pre-commit hooks..."
            pre-commit install
        else
            print_warning "pre-commit not found, skipping hook setup"
        fi
        
        print_status "Setup completed successfully!"
        ;;
    "test")
        print_status "Running tests..."
        if command -v pytest &> /dev/null; then
            pytest tests/ -v
        else
            print_warning "pytest not found, skipping tests"
        fi
        ;;
    "clean")
        print_status "Cleaning up generated files..."
        rm -rf checkpoints/* logs/* assets/evaluation/* assets/explainability/*
        print_status "Cleanup completed"
        ;;
    *)
        echo "X-ray Analysis System - Training and Evaluation Script"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  setup                    Set up the project environment"
        echo "  train [model]            Train a model (resnet18|efficientnet|vit|all)"
        echo "  eval [model]             Evaluate a model (resnet18|efficientnet|vit)"
        echo "  demo                     Launch interactive Streamlit demo"
        echo "  test                     Run unit tests"
        echo "  clean                    Clean up generated files"
        echo ""
        echo "Examples:"
        echo "  $0 setup                 # Initial setup"
        echo "  $0 train resnet18        # Train ResNet18 model"
        echo "  $0 train all            # Train all models"
        echo "  $0 eval resnet18         # Evaluate ResNet18 model"
        echo "  $0 demo                 # Launch demo"
        echo "  $0 test                 # Run tests"
        echo "  $0 clean                # Clean up"
        exit 1
        ;;
esac
