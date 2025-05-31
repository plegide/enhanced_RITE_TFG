import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics_data(analysis_method='edt_subpixel', synthesis_methods=['distance', 'or']):
    """Load metrics from CSV files."""
    metrics_dir = Path('results/metrics')
    print(f"\nLooking for metrics in: {metrics_dir.absolute()}")
    
    all_data = []
    for synthesis in synthesis_methods:
        method_name = f"{analysis_method}_{synthesis}"
        file_path = metrics_dir / f"metrics_{method_name}.csv"
        print(f"Checking for file: {file_path}")
        
        if file_path.exists():
            print(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path)
            df['Analysis_Method'] = analysis_method
            df['Synthesis_Method'] = synthesis
            df = df[df['Image'].str.contains('Image', na=False)]
            all_data.append(df)
        else:
            print(f"File not found: {file_path}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def load_all_methods_data():
    """Load metrics data for all possible method combinations."""
    analysis_methods = ['edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel']
    synthesis_methods = ['distance', 'or']
    
    image_data = []  # For individual image metrics
    stats_data = []  # For AVERAGE and STD rows
    metrics_dir = Path('results/metrics')
    
    for analysis in analysis_methods:
        for synthesis in synthesis_methods:
            method_name = f"{analysis}_{synthesis}"
            file_path = metrics_dir / f"metrics_{method_name.lower()}.csv"
            
            if file_path.exists():
                print(f"Loading data from: {file_path}")
                df = pd.read_csv(file_path)
                df['Analysis_Method'] = analysis
                df['Synthesis_Method'] = synthesis
                
                # Split data into metrics and statistics
                metrics_df = df[df['Image'].str.contains('Image', na=False)].copy()
                stats_df = df[df['Image'].isin(['AVERAGE', 'STD'])].copy()
                
                image_data.append(metrics_df)
                stats_data.append(stats_df)
    
    # Return both datasets
    return {
        'metrics': pd.concat(image_data, ignore_index=True) if image_data else pd.DataFrame(),
        'stats': pd.concat(stats_data, ignore_index=True) if stats_data else pd.DataFrame()
    }

def generate_roc_plots(stats_data):
    """Generate ROC space plots with mean points and std deviation ellipses."""
    plots_dir = Path('results/graphics/comparisons')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metric_pairs = [
        ('Precision', 'Recall'),
        ('Specificity', 'Recall')
    ]
    
    # Color map and methods setup
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    methods = []
    for analysis in ['edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel']:
        for synthesis in ['distance', 'or']:
            methods.append((analysis, synthesis))
    method_colors = dict(zip([f"{a}_{s}" for a,s in methods], colors))
    
    for x_metric, y_metric in metric_pairs:
        plt.figure(figsize=(10, 10))
        
        # Dictionary to track points for overlap detection
        points = {}
        
        # First plot all points and collect positions
        for analysis, synthesis in methods:
            method_stats = stats_data[
                (stats_data['Analysis_Method'] == analysis) & 
                (stats_data['Synthesis_Method'] == synthesis)
            ]
            
            if not method_stats.empty:
                avg_row = method_stats[method_stats['Image'] == 'AVERAGE']
                if not avg_row.empty:
                    x_mean = float(avg_row[x_metric].values[0])
                    y_mean = float(avg_row[y_metric].values[0])
                    method_name = f"{analysis}_{synthesis}"
                    
                    # Round coordinates for overlap detection
                    point_key = (round(x_mean, 4), round(y_mean, 4))
                    if point_key in points:
                        # Add method to overlapping points
                        if isinstance(points[point_key], list):
                            points[point_key].append(method_name)
                        else:
                            points[point_key] = [points[point_key], method_name]
                    else:
                        points[point_key] = method_name
                    
                    # Plot point
                    plt.scatter(x_mean, y_mean, c=[method_colors[method_name]], 
                              s=50, label=f"{analysis}\n{synthesis}")
                    
                    # Add method label for overlapping points
                    if point_key in points and isinstance(points[point_key], list):
                        plt.annotate('distance & or', 
                                    xy=(x_mean, y_mean),
                                    xytext=(5, 5),  # Fixed offset for all overlapping points
                                    textcoords='offset points',
                                    fontsize=8,
                                    bbox=dict(facecolor='white', 
                                            alpha=0.7,
                                            edgecolor='none'))
        
        # Then plot ellipses and axes
        for analysis, synthesis in methods:
            method_stats = stats_data[
                (stats_data['Analysis_Method'] == analysis) & 
                (stats_data['Synthesis_Method'] == synthesis)
            ]
            
            if not method_stats.empty:
                avg_row = method_stats[method_stats['Image'] == 'AVERAGE']
                std_row = method_stats[method_stats['Image'] == 'STD']
                
                if not avg_row.empty and not std_row.empty:
                    x_mean = float(avg_row[x_metric].values[0])
                    y_mean = float(avg_row[y_metric].values[0])
                    x_std = float(std_row[x_metric].values[0])
                    y_std = float(std_row[y_metric].values[0])
                    
                    method_name = f"{analysis}_{synthesis}"
                    color = method_colors[method_name]
                    
                    # Always plot both axes even if std is 0
                    plt.plot([x_mean - x_std, x_mean + x_std], 
                            [y_mean, y_mean],
                            color=color, alpha=0.2, linewidth=2)
                    plt.plot([x_mean, x_mean],
                            [y_mean - y_std, y_mean + y_std],
                            color=color, alpha=0.2, linewidth=2)
                    

                    angle = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = x_std * np.cos(angle) + x_mean
                    ellipse_y = y_std * np.sin(angle) + y_mean
                    plt.fill(ellipse_x, ellipse_y, color=color, alpha=0.2)
        
        # Add reference line
        plt.plot([0.65, 1.01], [0.95, 1.01], 'k--', alpha=0.3)
        
        # Create finer grid
        plt.grid(True, alpha=0.3)
        plt.minorticks_on()
        # Set number of minor ticks
        ax = plt.gca()
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        plt.grid(True, which='minor', alpha=0.15)
        
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.title(f'{y_metric} vs {x_metric} Space')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  title='Analysis Method\nSynthesis Method')
        plt.tight_layout()
        plt.savefig(plots_dir / f'roc_{x_metric.lower()}_{y_metric.lower()}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


def generate_comparison_plots(data):
    """Generate boxplots for Dice, Precision, Recall and Specificity."""
    plots_dir = Path('results/graphics/comparisons')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate boxplots for selected metrics
    metrics = ['Dice', 'Precision', 'Recall', 'Specificity']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)  # 2x2 layout
        sns.boxplot(
            data=data,
            x='Analysis_Method',
            y=metric,
            hue='Synthesis_Method',
            palette='Set2'
        )
        plt.title(f'{metric} Distribution')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_boxplots.png')
    plt.close()

def generate_all_graphics():
    """Generate all plots for method comparison."""
    print("\nLoading data for all methods...")
    data = load_all_methods_data()
    
    if data['metrics'].empty:
        print("No data found!")
        return
    
    print("\nGenerating comparison plots...")
    generate_comparison_plots(data['metrics'])  # Use metrics data for boxplots
    
    print("\nGenerating ROC space plots...")
    generate_roc_plots(data['stats'])  # Use stats data for ROC plots
    
    print("\nGenerating individual method plots...")
    # Use metrics data for individual plots
    for analysis in ['edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel']:
        for synthesis in ['distance', 'or']:
            method_data = data['metrics'][
                (data['metrics']['Analysis_Method'] == analysis) & 
                (data['metrics']['Synthesis_Method'] == synthesis)
            ]
            
            if method_data.empty:
                continue
            # Create method-specific directory
            method_dir = Path('results/graphics/methods') / f"{analysis}_{synthesis}"
            method_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate method-specific plots for selected metrics only
            metrics = ['Dice', 'Precision', 'Recall', 'Specificity']
            for metric in metrics:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=method_data, x=metric, kde=True)
                plt.title(f'{metric} Distribution for {analysis}_{synthesis}')
                plt.xlabel(metric)
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(method_dir / f'distribution_{metric}.png')
                plt.close()
    
    print("All plots generated successfully!")

