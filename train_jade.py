import os
import yaml
import argparse
import shutil
import json
from datetime import datetime
from JadeAssistant import JadeAssistant

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