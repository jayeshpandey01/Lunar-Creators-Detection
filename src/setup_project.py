import os
import shutil

def create_project_structure():
    """Create the project directory structure"""
    
    # Define the base directories
    directories = [
        'dataset/Dataset_01',
        'dataset/split_dataset/train',
        'dataset/split_dataset/val',
        'dataset/split_dataset/test',
        'src',
        'notebooks',
        'models',
        'logs',
        'checkpoints'
    ]
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def verify_structure():
    """Verify the project structure"""
    required_dirs = [
        'dataset',
        'src',
        'notebooks',
        'models',
        'logs',
        'checkpoints'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print("Missing directories:", missing_dirs)
        return False
    
    print("Project structure verified successfully!")
    return True

def setup_project():
    """Main setup function"""
    print("Setting up Mars Analysis project...")
    
    # Create directory structure
    create_project_structure()
    
    # Verify structure
    if verify_structure():
        print("\nProject setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your TIFF files in dataset/Dataset_01")
        print("2. Run split_single_tiff.py to create train/val/test splits")
        print("3. Start with the visualization notebook")
    else:
        print("\nProject setup failed. Please check the errors above.")

if __name__ == "__main__":
    setup_project() 