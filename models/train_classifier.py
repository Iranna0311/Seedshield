"""
SeedShield - MobileNetV2 Seed Classifier Training
Complete training script for seed classification model.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Progress bar
from tqdm import tqdm

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.model_config import (
        get_classifier_config, get_path_config,
        CLASSIFIER_CONFIG, PATH_CONFIG, CLASS_NAMES
    )
except ImportError:
    from model_config import (
        get_classifier_config, get_path_config,
        CLASSIFIER_CONFIG, PATH_CONFIG, CLASS_NAMES
    )


class SeedClassifier(nn.Module):
    """
    MobileNetV2-based seed classifier.
    Classifies individual seeds as REAL, FAKE, or LOW_QUALITY.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3):
        super(SeedClassifier, self).__init__()
        
        # Load pretrained MobileNetV2
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.base_model.classifier[1].in_features
        
        # Replace classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_features(self, x):
        """Extract features before the classifier."""
        x = self.base_model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class SeedClassifierTrainer:
    """
    Training pipeline for MobileNetV2 seed classifier.
    Handles class imbalance, augmentation, and comprehensive evaluation.
    """
    
    def __init__(self, config=None, paths=None):
        """
        Initialize the trainer.
        
        Args:
            config: ClassifierConfig instance
            paths: PathConfig instance
        """
        self.config = config or get_classifier_config()
        self.paths = paths or get_path_config()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def build_model(self):
        """Build the classifier model."""
        print(f"\nBuilding {self.config.model_name} classifier...")
        
        self.model = SeedClassifier(
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            dropout_rate=self.config.dropout_rate
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_data_loaders(self):
        """Setup data loaders with augmentation and class balancing."""
        print("\nSetting up data loaders...")
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        classification_dir = self.paths.classification_dataset
        
        train_dataset = ImageFolder(
            root=classification_dir / "train",
            transform=train_transform
        )
        
        val_dataset = ImageFolder(
            root=classification_dir / "val",
            transform=val_transform
        )
        
        test_dataset = ImageFolder(
            root=classification_dir / "test",
            transform=val_transform
        )
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Classes: {train_dataset.classes}")
        
        # Handle class imbalance with weighted sampler
        class_counts = np.bincount([label for _, label in train_dataset.samples])
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        print(f"  Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Store class weights for loss function
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        if self.model is None:
            self.build_model()
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        if self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate scheduler
        if self.config.lr_scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif self.config.lr_scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        
        print("\nTraining setup complete:")
        print(f"  Optimizer: {self.config.optimizer}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Scheduler: {self.config.lr_scheduler}")
    
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int = None):
        """
        Full training loop.
        
        Args:
            epochs: Number of training epochs
        
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        
        # Setup
        if self.train_loader is None:
            self.setup_data_loaders()
        self.setup_training()
        
        print(f"\n{'='*60}")
        print("TRAINING STARTED")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_model("best_classifier.pth")
                print(f"  ★ New best model saved (Val Acc: {val_acc:.2f}%)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"\nBest Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        # Save final model and history
        self.save_model("final_classifier.pth")
        self.save_history()
        self.plot_training_history()
        
        return self.history
    
    def evaluate(self, data_loader=None, split_name: str = "test"):
        """
        Evaluate the model and generate metrics.
        
        Args:
            data_loader: DataLoader to evaluate on
            split_name: Name of the split for reporting
        
        Returns:
            Dictionary with evaluation metrics
        """
        if data_loader is None:
            if self.test_loader is None:
                self.setup_data_loaders()
            data_loader = self.test_loader
        
        if self.model is None:
            self.load_model("best_classifier.pth")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"\nEvaluating on {split_name} set...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = 100 * np.mean(all_preds == all_labels)
        
        # Classification report
        class_names = self.config.class_names
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        print(f"\n{split_name.upper()} SET RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm, class_names, split_name)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist()
        }
    
    def plot_confusion_matrix(self, cm, class_names, split_name):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {split_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.paths.classifier_weights / f"confusion_matrix_{split_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_training_history(self):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        save_path = self.paths.classifier_weights / "training_history.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    
    def save_model(self, filename: str):
        """Save model weights."""
        save_path = self.paths.classifier_weights / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, filename: str):
        """Load model weights."""
        load_path = self.paths.classifier_weights / filename
        
        if not load_path.exists():
            print(f"Model file not found: {load_path}")
            return False
        
        if self.model is None:
            self.build_model()
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        print(f"Model loaded from: {load_path}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return True
    
    def save_history(self):
        """Save training history to JSON."""
        save_path = self.paths.classifier_weights / "training_history.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to: {save_path}")
    
    def export_to_onnx(self, filename: str = "classifier.onnx"):
        """Export model to ONNX format."""
        if self.model is None:
            self.load_model("best_classifier.pth")
        
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        
        export_path = self.paths.classifier_weights / filename
        
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {export_path}")
        return export_path


def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description='Train SeedShield Classifier')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export model to ONNX')
    
    args = parser.parse_args()
    
    trainer = SeedClassifierTrainer()
    
    if args.batch_size:
        trainer.config.batch_size = args.batch_size
    
    if args.evaluate_only:
        trainer.setup_data_loaders()
        trainer.load_model("best_classifier.pth")
        trainer.evaluate()
    else:
        # Train the model
        trainer.train(epochs=args.epochs)
        
        # Evaluate if requested
        if args.evaluate:
            trainer.evaluate()
        
        # Export if requested
        if args.export_onnx:
            trainer.export_to_onnx()


if __name__ == "__main__":
    main()
