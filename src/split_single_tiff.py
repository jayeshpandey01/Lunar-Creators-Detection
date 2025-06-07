"""
Script to split a single TIFF dataset into train and test sets
Specifically for dataset/Dataset_01/M175124932 structure
with random stratified splitting
"""

import os
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SingleTiffSplitter:
    def __init__(self, input_dir, output_base_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Initialize SingleTiffSplitter with random splitting
        
        Args:
            input_dir (str): Directory containing M175124932 TIFF tiles
            output_base_dir (str): Base directory to save split dataset
            train_ratio (float): Ratio of training data (0-1)
            val_ratio (float): Ratio of validation data (0-1)
            seed (int): Random seed for reproducibility
        """
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directories
        self.train_dir = os.path.join(output_base_dir, 'train', 'M175124932LR')
        self.val_dir = os.path.join(output_base_dir, 'val', 'M175124932LR')
        self.test_dir = os.path.join(output_base_dir, 'test', 'M175124932LR')
        
    def create_directories(self):
        """Create necessary directories for train/val/test splits"""
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            
    def get_tile_position(self, filename):
        """Extract tile position from filename for stratified splitting"""
        try:
            # Assuming filename format: tile_Y_X.tiff
            parts = filename.split('_')
            y, x = int(parts[1]), int(parts[2].split('.')[0])
            return y, x
        except:
            return None
            
    def copy_files(self, files, src_dir, dst_dir):
        """Copy files from source to destination directory"""
        for file in tqdm(files, desc=f"Copying to {os.path.basename(dst_dir)}"):
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.copy2(src_path, dst_path)
            
    def split_dataset(self):
        """Split the dataset into train, validation, and test sets with stratification"""
        print("Starting dataset splitting...")
        
        # Create directory structure
        self.create_directories()
        
        # Get all TIFF images and their positions
        images = [f for f in os.listdir(self.input_dir) 
                 if f.endswith(('.tiff', '.tif'))]
        
        if not images:
            raise ValueError(f"No TIFF images found in {self.input_dir}")
        
        # Create position-based groups for stratification
        image_positions = [(img, self.get_tile_position(img)) for img in images]
        image_positions = [(img, pos) for img, pos in image_positions if pos is not None]
        
        # Sort by position for better distribution
        image_positions.sort(key=lambda x: (x[1][0], x[1][1]))
        images = [img for img, _ in image_positions]
        
        print(f"Found {len(images)} TIFF images")
        
        # Perform stratified split
        train_images, temp_images = train_test_split(
            images,
            train_size=self.train_ratio,
            random_state=self.seed,
            shuffle=True
        )
        
        # Split temporary set into validation and test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_images, test_images = train_test_split(
            temp_images,
            train_size=val_ratio_adjusted,
            random_state=self.seed,
            shuffle=True
        )
        
        # Copy files to respective directories
        self.copy_files(train_images, self.input_dir, self.train_dir)
        self.copy_files(val_images, self.input_dir, self.val_dir)
        self.copy_files(test_images, self.input_dir, self.test_dir)
        
        # Print statistics
        print("\nDataset split statistics:")
        print(f"Total images: {len(images)}")
        print(f"Training images: {len(train_images)} ({len(train_images)/len(images)*100:.1f}%)")
        print(f"Validation images: {len(val_images)} ({len(val_images)/len(images)*100:.1f}%)")
        print(f"Testing images: {len(test_images)} ({len(test_images)/len(images)*100:.1f}%)")
        
        # Analyze spatial distribution
        self.analyze_spatial_distribution(train_images, "Training")
        self.analyze_spatial_distribution(val_images, "Validation")
        self.analyze_spatial_distribution(test_images, "Testing")
        
    def analyze_spatial_distribution(self, images, split_name):
        """Analyze the spatial distribution of tiles in each split"""
        positions = [self.get_tile_position(img) for img in images]
        positions = [pos for pos in positions if pos is not None]
        
        if positions:
            y_coords, x_coords = zip(*positions)
            print(f"\n{split_name} set spatial distribution:")
            print(f"Y-range: {min(y_coords)} to {max(y_coords)}")
            print(f"X-range: {min(x_coords)} to {max(x_coords)}")
        
    def verify_split(self):
        """Verify the split was successful and balanced"""
        print("\nVerifying split results:")
        
        for split_dir, split_name in [(self.train_dir, 'Training'),
                                    (self.val_dir, 'Validation'),
                                    (self.test_dir, 'Testing')]:
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) 
                        if f.endswith(('.tiff', '.tif'))]
                count = len(files)
                
                # Analyze spatial distribution in verification
                positions = [self.get_tile_position(f) for f in files]
                positions = [p for p in positions if p is not None]
                
                print(f"\n{split_name} set:")
                print(f"Total images: {count}")
                if positions:
                    y_coords, x_coords = zip(*positions)
                    print(f"Spatial coverage:")
                    print(f"  Y-range: {min(y_coords)} to {max(y_coords)}")
                    print(f"  X-range: {min(x_coords)} to {max(x_coords)}")
            else:
                print(f"Warning: {split_name} directory not found!")

def main():
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    input_dir = os.path.join(base_dir, 'dataset', 'Dataset_01', 'M175124932LR')
    output_base_dir = os.path.join(base_dir, "dataset", "split_dataset")
    
    # Verify input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print("Please ensure the correct directory structure exists")
        return

    # Initialize and run splitter
    splitter = SingleTiffSplitter(
        input_dir=input_dir,
        output_base_dir=output_base_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    try:
        splitter.split_dataset()
        splitter.verify_split()
        print("\nDataset splitting completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset splitting: {str(e)}")

if __name__ == "__main__":
    main() 