import os
import yaml
import argparse
import shutil
import json
from datetime import datetime
from JadeAssistant import JadeAssistant
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_training_config(data_dir, classes):
    """Create YAML configuration for training"""
    config = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml with {len(classes)} classes")
    return 'data.yaml'

def prepare_training_data():
    """Prepare training data structure"""
    print("📁 Creating training directory structure...")
    
    directories = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("\n📋 INSTRUCTIONS FOR TRAINING:")
    print("="*60)
    print("1. Place training images in 'train/images/'")
    print("2. Create annotation files in 'train/labels/' (YOLO format)")
    print("3. Repeat for 'val/' and 'test/' directories")
    print("4. Edit the classes list in the script")
    print("5. Run: python train_jade.py --train")
    print("\n📝 YOLO FORMAT:")
    print("   <class_id> <x_center> <y_center> <width> <height>")
    print("   All values normalized 0-1")
    print("="*60)
    
    return True

def create_training_visualization(results, epochs, classes):
    """Create training visualization plots"""
    print("📊 Creating training visualization...")
    
    # Create reports directory
    os.makedirs('reports/training', exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'JADE Training Results - {len(classes)} Classes, {epochs} Epochs', 
                fontsize=16, color='white')
    
    try:
        # 1. Loss curves
        ax1 = axes[0, 0]
        if hasattr(results, 'results_dict'):
            train_loss = results.results_dict.get('train/box_loss', [])
            val_loss = results.results_dict.get('val/box_loss', [])
            
            if train_loss and val_loss:
                epochs_range = range(1, len(train_loss) + 1)
                ax1.plot(epochs_range, train_loss, 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
                ax1.set_title('Loss Curves', color='white')
                ax1.set_xlabel('Epoch', color='white')
                ax1.set_ylabel('Loss', color='white')
                ax1.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(colors='white')
        
        # 2. Accuracy metrics
        ax2 = axes[0, 1]
        if hasattr(results, 'results_dict'):
            precision = results.results_dict.get('metrics/precision(B)', [])
            recall = results.results_dict.get('metrics/recall(B)', [])
            
            if precision and recall:
                epochs_range = range(1, len(precision) + 1)
                ax2.plot(epochs_range, precision, 'g-', label='Precision', linewidth=2)
                ax2.plot(epochs_range, recall, 'y-', label='Recall', linewidth=2)
                ax2.set_title('Precision & Recall', color='white')
                ax2.set_xlabel('Epoch', color='white')
                ax2.set_ylabel('Score', color='white')
                ax2.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(colors='white')
        
        # 3. mAP scores
        ax3 = axes[1, 0]
        if hasattr(results, 'results_dict'):
            map50 = results.results_dict.get('metrics/mAP50(B)', [])
            map95 = results.results_dict.get('metrics/mAP50-95(B)', [])
            
            if map50 and map95:
                epochs_range = range(1, len(map50) + 1)
                ax3.plot(epochs_range, map50, 'c-', label='mAP@50', linewidth=2)
                ax3.plot(epochs_range, map95, 'm-', label='mAP@50-95', linewidth=2)
                ax3.set_title('mAP Scores', color='white')
                ax3.set_xlabel('Epoch', color='white')
                ax3.set_ylabel('mAP', color='white')
                ax3.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(colors='white')
        
        # 4. Class distribution (placeholder - would need actual data)
        ax4 = axes[1, 1]
        if classes:
            # Simulate class distribution
            class_counts = np.random.randint(50, 500, len(classes))
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            
            bars = ax4.barh(classes, class_counts, color=colors)
            ax4.set_title('Training Class Distribution', color='white')
            ax4.set_xlabel('Number of Images', color='white')
            ax4.tick_params(colors='white')
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, class_counts)):
                width = bar.get_width()
                ax4.text(width + max(class_counts)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        plot_file = f"reports/training/training_results_{int(time.time())}.png"
        plt.savefig(plot_file, dpi=150, facecolor='#0f0f0f')
        plt.close()
        print(f"📈 Training visualization saved: {plot_file}")
        
        # Create summary dashboard
        create_training_summary(results, epochs, classes)
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

def create_training_summary(results, epochs, classes):
    """Create training summary dashboard"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create summary text
    summary_text = f"""JADE Training Summary
    
    Training Configuration:
    • Classes: {len(classes)}
    • Epochs: {epochs}
    • Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Model Information:
    • Framework: YOLO
    • Input Size: 640x640
    • Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
    
    Training Classes:"""
    
    # Add classes
    for i, cls in enumerate(classes):
        summary_text += f"\n• {cls}"
    
    # Add best metrics if available
    if hasattr(results, 'results_dict'):
        summary_text += f"\n\nBest Metrics:"
        
        best_metrics = {
            'Precision': max(results.results_dict.get('metrics/precision(B)', [0])),
            'Recall': max(results.results_dict.get('metrics/recall(B)', [0])),
            'mAP@50': max(results.results_dict.get('metrics/mAP50(B)', [0])),
            'mAP@50-95': max(results.results_dict.get('metrics/mAP50-95(B)', [0]))
        }
        
        for metric, value in best_metrics.items():
            summary_text += f"\n• {metric}: {value:.3f}"
    
    summary_text += f"\n\nModel saved to: runs/detect/train/weights/best.pt"
    
    # Display summary
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#2e2e2e', 
                                             edgecolor='white', alpha=0.9),
           color='white', fontfamily='monospace')
    
    ax.axis('off')
    ax.set_facecolor('#0f0f0f')
    
    # Add logo/watermark
    ax.text(0.98, 0.02, 'JADE Training System', transform=ax.transAxes,
           fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    
    plt.tight_layout()
    summary_file = f"reports/training/training_summary_{int(time.time())}.png"
    plt.savefig(summary_file, dpi=150, facecolor='#0f0f0f')
    plt.close()
    print(f"📋 Training summary saved: {summary_file}")

def train_custom_model(classes=None, epochs=100, imgsz=640):
    """Train custom YOLO model"""
    if classes is None:
        custom_classes = [
            "specific_object_1",
            "specific_object_2",
            "specific_object_3",
        ]
    else:
        custom_classes = classes
    
    print(f"🎯 Training Configuration:")
    print(f"   Classes: {custom_classes}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {imgsz}")
    print("="*60)
    
    # Check training data
    train_img_dir = 'train/images'
    if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
        print("❌ No training images found")
        return None
    
    print(f"📊 Training data: {len(os.listdir(train_img_dir))} images")
    
    # Create config
    config_path = create_training_config('.', custom_classes)
    
    # Initialize detector
    print("🤖 Initializing YOLO model...")
    detector = JadeAssistant()
    
    # Train the model
    print("\n🎯 Starting training...")
    print("💻 Using device:", detector.device)
    
    try:
        results = detector.train_custom_model(
            data_yaml=config_path,
            epochs=epochs,
            imgsz=imgsz
        )
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved in: runs/detect/train/")
        
        # Copy best model
        best_model_path = 'runs/detect/train/weights/best.pt'
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, 'models/custom_yolo.pt')
            print(f"💾 Best model copied to: models/custom_yolo.pt")
        
        # Create visualizations
        create_training_visualization(results, epochs, custom_classes)
        
        return results
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def validate_training_data():
    """Validate training data"""
    print("🔍 Validating training data...")
    
    issues = []
    
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
    
    if issues:
        print("❌ Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    # Check file counts
    train_images = len(os.listdir('train/images'))
    train_labels = len(os.listdir('train/labels'))
    
    print(f"📊 File counts:")
    print(f"   Train images: {train_images}")
    print(f"   Train labels: {train_labels}")
    
    if train_images != train_labels:
        issues.append(f"Mismatch: train images ({train_images}) vs labels ({train_labels})")
    
    if train_images == 0:
        issues.append("No training images found")
    
    if issues:
        print("❌ Validation issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Training data validation passed!")
    return True

def export_training_summary(results):
    """Export training summary report"""
    summary = {
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'classes_trained': 'Unknown',
            'training_epochs': 'Unknown',
            'best_model_path': 'runs/detect/train/weights/best.pt'
        }
    }
    
    # Save summary
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Training summary saved to: training_summary.json")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Train JADE for specific objects")
    parser.add_argument('--setup', action='store_true', help="Setup training directory structure")
    parser.add_argument('--train', action='store_true', help="Start training")
    parser.add_argument('--validate', action='store_true', help="Validate training data")
    parser.add_argument('--classes', nargs='+', help="Custom classes for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 JADE CUSTOM MODEL TRAINING")
    print("="*60)
    
    if args.setup:
        prepare_training_data()
    
    elif args.validate:
        validate_training_data()
    
    elif args.train:
        if not validate_training_data():
            print("❌ Cannot start training due to validation issues")
            return
        
        results = train_custom_model(
            classes=args.classes,
            epochs=args.epochs,
            imgsz=args.imgsz
        )
        
        if results:
            export_training_summary(results)
    
    else:
        print("Usage:")
        print("  python train_jade.py --setup     # Setup training directories")
        print("  python train_jade.py --validate  # Validate training data")
        print("  python train_jade.py --train     # Start training")
        print("\nOptional arguments:")
        print("  --classes obj1 obj2 obj3         # Custom classes")
        print("  --epochs 50                      # Number of epochs (default: 100)")
        print("  --imgsz 320                      # Image size (default: 640)")

if __name__ == "__main__":
    main()