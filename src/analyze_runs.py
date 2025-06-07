from visualization_utils import WandBVisualizer

def analyze_experiments():
    visualizer = WandBVisualizer()
    
    # Compare recent runs automatically
    visualizer.compare_recent_runs(
        project_name='tiff-classification',
        entity='jayeshpandey020',
        metric='accuracy',
        limit=5,  # number of recent runs to compare
        save_path='accuracy_comparison.png'
    )
    
    visualizer.compare_recent_runs(
        project_name='tiff-classification',
        entity='jayeshpandey020',
        metric='loss',
        limit=5,
        save_path='loss_comparison.png'
    )

if __name__ == '__main__':
    analyze_experiments() 