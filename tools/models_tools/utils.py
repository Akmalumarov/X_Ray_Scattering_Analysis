"""
Utility functions for X-Ray models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

def plot_model_decomposition(model, x: np.ndarray, ax: Optional[plt.Axes] = None, 
                           show_components: bool = True) -> plt.Axes:
    """
    Plot model with all its components.
    
    Args:
        model: Model or CompositeModel
        x: X-values array
        ax: Matplotlib axes (creates new if None)
        show_components: Whether to show individual components
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot total model
    y_total = model(x)
    ax.plot(x, y_total, 'k-', linewidth=3, label='Total model')
    
    # Plot components if available
    if show_components and hasattr(model, 'models'):
        # Get color cycle for consistent coloring
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i, component in enumerate(model.models):
            y_component = component(x)
            color = colors[i % len(colors)]
            ax.plot(x, y_component, '--', color=color, 
                   linewidth=2, alpha=0.8, 
                   label=f'{component.__class__.__name__}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax

def get_model_components(model) -> Dict[str, np.ndarray]:
    """
    Get all components of a model.
    
    Args:
        model: Model or CompositeModel
        
    Returns:
        Dictionary with component names and values
    """
    components = {}
    
    if hasattr(model, 'models'):
        # Composite model
        for i, component in enumerate(model.models):
            comp_name = component.__class__.__name__
            if hasattr(component, 'prefix') and component.prefix:
                comp_name = f"{component.prefix}{comp_name}"
            components[comp_name] = component
    else:
        # Single model
        components[model.__class__.__name__] = model
    
    return components

def quick_plot(model, x: np.ndarray, filename: Optional[str] = None) -> None:
    """
    Quick plot of model with components.
    
    Args:
        model: Model to plot
        x: X-values array
        filename: If provided, save figure to file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Just total model
    y_total = model(x)
    axes[0].plot(x, y_total, 'b-', linewidth=2)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Total Model', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: With components
    plot_model_decomposition(model, x, axes[1])
    axes[1].set_title('Model with Components', fontsize=14)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()
    
    # Print summary
    components = get_model_components(model)
    print(f"Model has {len(components)} component(s):")
    for name, component in components.items():
        print(f"  - {name}: {component}")

def compare_models(models_dict: Dict[str, object], x: np.ndarray, 
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Compare multiple models on the same plot.
    
    Args:
        models_dict: Dictionary with model names as keys
        x: X-values array
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (name, model) in enumerate(models_dict.items()):
        color = colors[i % len(colors)]
        y = model(x)
        ax.plot(x, y, '-', color=color, linewidth=2, label=name)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax

def create_model_summary(model) -> str:
    """
    Create a text summary of the model.
    
    Args:
        model: Model to summarize
        
    Returns:
        Summary string
    """
    summary = []
    summary.append("=" * 50)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 50)
    
    if hasattr(model, '__class__'):
        summary.append(f"Type: {model.__class__.__name__}")
    
    if hasattr(model, 'name'):
        summary.append(f"Name: {model.name}")
    
    components = get_model_components(model)
    summary.append(f"\nComponents ({len(components)}):")
    
    for name, component in components.items():
        summary.append(f"\n  {name}:")
        if hasattr(component, 'params'):
            for param, value in component.params.items():
                summary.append(f"    {param}: {value}")
    
    summary.append("=" * 50)
    
    return "\n".join(summary)

def save_model_data(model, x: np.ndarray, filename: str) -> None:
    """
    Save model data to CSV file.
    
    Args:
        model: Model to evaluate
        x: X-values array
        filename: Output CSV filename
    """
    import pandas as pd
    
    data = {'x': x}
    y_total = model(x)
    data['y_total'] = y_total
    
    # Add components
    if hasattr(model, 'models'):
        for i, component in enumerate(model.models):
            comp_name = component.__class__.__name__
            if hasattr(component, 'prefix') and component.prefix:
                comp_name = f"{component.prefix}_{comp_name}"
            data[f'y_{comp_name}'] = component(x)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Model data saved to {filename}")

def plot(model, x):
    """Plot model with all components in one line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Total
    y_total = model(x)
    ax.plot(x, y_total, 'k-', linewidth=3, label='Total')
    
    # Components
    if hasattr(model, 'models'):
        for comp in model.models:
            ax.plot(x, comp(x), '--', linewidth=2, label=comp.__class__.__name__)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def decompose(model):
    """Get all components as list"""
    return model.models if hasattr(model, 'models') else [model]
    
# Short aliases for convenience
plot = plot_model_decomposition
summary = create_model_summary
compare = compare_models