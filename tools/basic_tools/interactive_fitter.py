окей, считаем это правильной официальной версией:
# tools/basic_tools/interactive_fitter.py
"""
Interactive fitter with Plotly for data exploration
"""

import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

class InteractiveFitter:
    """Interactive curve fitting with Plotly visualization"""
    
    def __init__(self, x_data, y_data, model_func, initial_params=None):
        self.x_data = x_data
        self.y_data = y_data
        self.model_func = model_func
        self.initial_params = initial_params or [1.0] * (model_func.__code__.co_argcount - 1)
        
        # Fitting bounds
        self.fit_min = np.min(x_data)
        self.fit_max = np.max(x_data)
        
        # Fit results
        self.popt = None
        self.perr = None
        
        # Create widgets and display initial plot
        self._create_widgets()
        self._display_initial_plot()
        
    def _create_widgets(self):
        """Create interactive widgets"""
        # Sliders for bounds
        self.min_slider = widgets.FloatSlider(
            value=self.fit_min,
            min=np.min(self.x_data),
            max=np.max(self.x_data),
            description='Min bound:',
            continuous_update=True
        )
        
        self.max_slider = widgets.FloatSlider(
            value=self.fit_max,
            min=np.min(self.x_data),
            max=np.max(self.x_data),
            description='Max bound:',
            continuous_update=True
        )
        
        # Fit button (optional, but keep for manual control)
        self.fit_button = widgets.Button(
            description='Fit curve',
            button_style='primary'
        )
        
        # Auto-fit checkbox
        self.auto_fit_checkbox = widgets.Checkbox(
            value=True,
            description='Auto-fit on bounds change'
        )
        
        # Output for plot
        self.plot_output = widgets.Output()
        
        # Output for text results
        self.results_output = widgets.Output()
        
        # Connect events
        self.min_slider.observe(self._update_bounds, names='value')
        self.max_slider.observe(self._update_bounds, names='value')
        self.fit_button.on_click(self._perform_fit)
        self.auto_fit_checkbox.observe(self._update_auto_fit, names='value')
        
    def _update_auto_fit(self, change):
        """Update auto-fit setting"""
        self.auto_fit = change['new']
        
    def _display_initial_plot(self):
        """Display initial plot with experimental data"""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            self._create_plot()
    
    def _update_bounds(self, change):
        """Update fitting bounds and refresh plot"""
        self.fit_min = self.min_slider.value
        self.fit_max = self.max_slider.value
        
        # Perform auto-fit if enabled
        if hasattr(self, 'auto_fit') and self.auto_fit and self.popt is not None:
            # Check if we have enough points for fitting
            mask = (self.x_data >= self.fit_min) & (self.x_data <= self.fit_max)
            if np.sum(mask) >= len(self.initial_params):
                self._perform_fit(None)
            else:
                # Just refresh plot without fitting
                self._refresh_plot()
        else:
            self._refresh_plot()
    
    def _refresh_plot(self):
        """Refresh plot with current bounds"""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            self._create_plot()
    
    def _create_plot(self):
        """Create Plotly figure with current data"""
        # Create figure
        fig = go.Figure()
        
        # Add experimental data
        fig.add_trace(go.Scatter(
            x=self.x_data,
            y=self.y_data,
            mode='markers',
            name='Experimental data',
            marker=dict(size=8, color='black'),
            hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
        
        # Add fitting region
        fig.add_vrect(
            x0=self.fit_min, x1=self.fit_max,
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Fitting region",
            annotation_position="top left"
        )
        
        # Add fit if available
        if self.popt is not None:
            x_fine = np.linspace(self.fit_min, self.fit_max, 200)
            y_fit = self.model_func(x_fine, *self.popt)
            
            fig.add_trace(go.Scatter(
                x=x_fine,
                y=y_fit,
                mode='lines',
                name='Fit',
                line=dict(color='red', width=3),
                hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
            ))
            
            # Extrapolation
            x_ext = np.linspace(np.min(self.x_data), np.max(self.x_data), 300)
            y_ext = self.model_func(x_ext, *self.popt)
            
            fig.add_trace(go.Scatter(
                x=x_ext,
                y=y_ext,
                mode='lines',
                name='Extrapolation',
                line=dict(color='red', width=3, dash='dash'),
                opacity=0.5
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Curve Fitting',
            xaxis_title='X',
            yaxis_title='Y',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        # Show figure
        fig.show()
    
    def _perform_fit(self, b):
        """Perform curve fitting and update plot"""
        # Select data within bounds
        mask = (self.x_data >= self.fit_min) & (self.x_data <= self.fit_max)
        x_fit = self.x_data[mask]
        y_fit = self.y_data[mask]
        
        if len(x_fit) < len(self.initial_params):
            # Just refresh plot without fitting
            self._refresh_plot()
            with self.results_output:
                self.results_output.clear_output(wait=True)
                print(f"Not enough points for fitting. Need at least {len(self.initial_params)}, have {len(x_fit)}")
            return
        
        try:
            # Fit
            self.popt, pcov = curve_fit(
                self.model_func, x_fit, y_fit, 
                p0=self.initial_params, maxfev=5000
            )
            self.perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
            
            # Update plot with fit
            self._refresh_plot()
            
            # Display results
            with self.results_output:
                self.results_output.clear_output(wait=True)
                self._print_results()
            
        except Exception as e:
            with self.results_output:
                print(f"Fitting error: {e}")
    
    def _print_results(self):
        """Print fitting results"""
        print("=" * 50)
        print("FITTING RESULTS")
        print("=" * 50)
        print(f"Fitting range: [{self.fit_min:.4g}, {self.fit_max:.4g}]")
        print(f"Points in fit: {np.sum((self.x_data >= self.fit_min) & (self.x_data <= self.fit_max))}")
        print("-" * 50)
        
        # Handle case when perr is None
        if self.perr is not None:
            for i, (param, err) in enumerate(zip(self.popt, self.perr)):
                print(f"Parameter {i}: {param:.6g} ± {err:.6g}")
        else:
            for i, param in enumerate(self.popt):
                print(f"Parameter {i}: {param:.6g}")
        
        print("=" * 50)
    
    def show(self):
        """Display the interface"""
        # Initialize auto-fit
        self.auto_fit = self.auto_fit_checkbox.value
        
        # Arrange widgets
        controls = widgets.VBox([
            widgets.HBox([self.min_slider, self.max_slider]),
            widgets.HBox([self.auto_fit_checkbox, self.fit_button]),
            self.plot_output,
            self.results_output
        ])
        
        display(controls)

    def get_fit_results(self):
        """Return fitting results as a dictionary"""
        if self.popt is None:
            return None
        
        results = {
            'parameters': self.popt,
            'errors': self.perr,
            'fit_range': (self.fit_min, self.fit_max),
            'points_in_fit': np.sum((self.x_data >= self.fit_min) & (self.x_data <= self.fit_max)),
            'model_function': self.model_func
        }
        
        # Calculate R-squared if we have a fit
        if self.popt is not None:
            mask = (self.x_data >= self.fit_min) & (self.x_data <= self.fit_max)
            x_fit = self.x_data[mask]
            y_fit = self.y_data[mask]
            y_pred = self.model_func(x_fit, *self.popt)
            
            # R-squared calculation
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            if ss_tot != 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = np.nan
            
            results['r_squared'] = r_squared
            results['residuals'] = y_fit - y_pred
        
        return results
    
    def get_fitted_function(self):
        """Return a function with fitted parameters"""
        if self.popt is None:
            return None
        
        return lambda x: self.model_func(x, *self.popt)