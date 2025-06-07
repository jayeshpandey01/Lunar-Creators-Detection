"""
Script to split TIFF dataset into train and test sets
Maintains class structure and handles data splitting with proper ratios
"""

import os
import shutil
import random
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

class DatasetSplitter:
    def __init__(self, input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Initialize DatasetSplitter
        
        Args:
            input_dir (str): Directory containing class folders with images
            output_dir (str): Directory to save split dataset
            train_ratio (float): Ratio of training data (0-1)
            val_ratio (float): Ratio of validation data (0-1)
            seed (int): Random seed for reproducibility
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.seed = seed
        random.seed(seed)
        
        # Create output directories
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        self.test_dir = os.path.join(output_dir, 'test')
        
    def create_directories(self):
        """Create necessary directories for train/val/test splits"""
        # Create main output directories
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            
        # Create class subdirectories
        class_dirs = [d for d in os.listdir(self.input_dir) 
                     if os.path.isdir(os.path.join(self.input_dir, d))]
        
        for class_dir in class_dirs:
            os.makedirs(os.path.join(self.train_dir, class_dir))
            os.makedirs(os.path.join(self.val_dir, class_dir))
            os.makedirs(os.path.join(self.test_dir, class_dir))
            
    def copy_files(self, files, src_dir, dst_dir, class_name):
        """Copy files from source to destination directory"""
        for file in files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, class_name, file)
            shutil.copy2(src_path, dst_path)
            
    def split_dataset(self):
        """Split the dataset into train, validation, and test sets"""
        print("Starting dataset splitting...")
        
        # Create directory structure
        self.create_directories()
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.input_dir) 
                     if os.path.isdir(os.path.join(self.input_dir, d))]
        
        # Process each class
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            class_path = os.path.join(self.input_dir, class_dir)
            
            # Get all images in the class directory
            images = [f for f in os.listdir(class_path) 
                     if f.endswith(('.tiff', '.tif'))]
            
            if not images:
                print(f"Warning: No images found in {class_path}")
                continue
                
            # Split into train and temporary set
            train_images, temp_images = train_test_split(
                images,
                train_size=self.train_ratio,
                random_state=self.seed
            )
            
            # Split temporary set into validation and test
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_images, test_images = train_test_split(
                temp_images,
                train_size=val_ratio_adjusted,
                random_state=self.seed
            )
            
            # Copy files to respective directories
            self.copy_files(train_images, class_path, self.train_dir, class_dir)
            self.copy_files(val_images, class_path, self.val_dir, class_dir)
            self.copy_files(test_images, class_path, self.test_dir, class_dir)
            
            # Print statistics
            print(f"\nClass {class_dir} statistics:")
            print(f"Total images: {len(images)}")
            print(f"Training images: {len(train_images)}")
            print(f"Validation images: {len(val_images)}")
            print(f"Testing images: {len(test_images)}")
            
    def verify_split(self):
        """Verify the split was successful"""
        print("\nVerifying split results:")
        
        for split_dir, split_name in [(self.train_dir, 'Training'),
                                    (self.val_dir, 'Validation'),
                                    (self.test_dir, 'Testing')]:
            total_images = 0
            class_counts = {}
            
            for class_dir in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_dir)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.endswith(('.tiff', '.tif'))])
                    class_counts[class_dir] = count
                    total_images += count
            
            print(f"\n{split_name} set:")
            print(f"Total images: {total_images}")
            print("Class distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} images")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing class folders with images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Ratio of training data (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='Ratio of validation data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Train and validation ratios sum should be less than 1")
    
    # Initialize and run splitter
    splitter = DatasetSplitter(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    splitter.split_dataset()
    splitter.verify_split()

if __name__ == "__main__":
    main() 