import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
warnings.filterwarnings('ignore')
from datasets.ostrinia import Ostrinia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
warnings.filterwarnings('ignore')

class SpatiotemporalVisualizer:
    def __init__(self, df, mask, output_dir='plots', var_name=None):
        """
        Initialize the visualizer with data and mask.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with nodes as columns and dates as index
        mask : numpy.ndarray or pandas.DataFrame
            Binary mask (1 for valid data, 0 for invalid)
        output_dir : str
            Base directory to save plots (default: 'plots')
        var_name : str
            Name of the variable being visualized (used in titles and subfolder)
        """
        self.df = df
        self.mask = mask if isinstance(mask, pd.DataFrame) else pd.DataFrame(mask, 
                                                                             index=df.index, 
                                                                             columns=df.columns)
        self.var_name = var_name if var_name else "Signal"
        
        # Create output directory with variable subfolder
        if var_name:
            self.output_dir = os.path.join(output_dir, var_name)
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create masked data where invalid points are NaN
        self.masked_df = self.df.where(self.mask == 1)
        
    def _format_time_axis(self, ax, rotation=45):
        """
        Format time axis with better date labels.
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to format
        rotation : int
            Rotation angle for labels
        """
        # Convert index to datetime if it's not already
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                time_index = pd.to_datetime(self.df.index)
            except:
                # If conversion fails, just use simple formatting
                # Reduce number of labels shown
                n_labels = min(10, len(self.df.index))
                indices = np.linspace(0, len(self.df.index)-1, n_labels, dtype=int)
                ax.set_xticks(indices)
                ax.set_xticklabels([str(self.df.index[i])[:10] for i in indices], 
                                 rotation=rotation, ha='right')
                return
        else:
            time_index = self.df.index
        
        # Determine appropriate date format based on time range
        time_range = (time_index[-1] - time_index[0]).days
        
        if time_range <= 7:  # Week or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        elif time_range <= 30:  # Month or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        elif time_range <= 365:  # Year or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:  # More than a year
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Rotate labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')
        
    def _add_year_lines(self, ax, alpha=0.3):
        """
        Add vertical lines at year boundaries.
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add lines to
        alpha : float
            Transparency of the year lines
        """
        # Convert index to datetime if needed
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                time_index = pd.to_datetime(self.df.index)
            except:
                # If conversion fails, skip year lines
                return
        else:
            time_index = self.df.index
            
        # Get unique years in the data
        years = time_index.year.unique()
        
        if len(years) > 1:
            # Get the start and end dates
            start_date = time_index[0]
            end_date = time_index[-1]
            
            # For each year, check if January 1st falls within our data range
            for year in years:
                year_start = pd.Timestamp(f'{year}-01-01')
                
                # Skip if January 1st is before our data starts
                if year_start <= start_date:
                    continue
                    
                # Skip if January 1st is after our data ends
                if year_start >= end_date:
                    continue
                
                # Draw the line at January 1st
                ax.axvline(x=year_start, color='gray', linestyle='--', 
                          alpha=alpha, linewidth=1)
                
                # Add year label
                ax.text(year_start, ax.get_ylim()[1], str(year), 
                       rotation=0, ha='left', va='bottom', 
                       fontsize=8, color='gray', alpha=0.7)
                               
    def _add_year_lines_heatmap(self, ax, time_index, alpha=0.5):
        """
        Add vertical lines at year boundaries for heatmaps.
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add lines to
        time_index : pandas.DatetimeIndex
            Datetime index
        alpha : float
            Transparency of the year lines
        """
        # Get unique years in the data
        years = time_index.year.unique()
        
        if len(years) > 1:
            # Get the start and end dates
            start_date = time_index[0]
            end_date = time_index[-1]
            
            # For each year, find January 1st
            for year in years:
                year_start = pd.Timestamp(f'{year}-01-01')
                
                # Skip if January 1st is before our data starts
                if year_start < start_date:
                    continue
                    
                # Skip if this is the first data point (no line needed at the very start)
                if year_start == start_date:
                    continue
                
                # Skip if January 1st is after our data ends
                if year_start > end_date:
                    continue
                
                # Find the closest index position to January 1st
                # Calculate the position as a fraction of the total time range
                time_fraction = (year_start - start_date) / (end_date - start_date)
                x_position = time_fraction * len(time_index)
                
                # Draw vertical line
                ax.axvline(x=x_position, color='white', linestyle='--', 
                          alpha=alpha, linewidth=1.5)
                
                # Add year label at the top
                ax.text(x_position, ax.get_ylim()[1]*0.98, str(year), 
                       rotation=0, ha='left', va='top', 
                       fontsize=10, color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
    # def plot_heatmap(self, figsize=(12, 8), cmap='viridis', show_mask=False):
    #     """
    #     Create a heatmap of the spatiotemporal data.
        
    #     Parameters:
    #     -----------
    #     figsize : tuple
    #         Figure size (width, height)
    #     cmap : str
    #         Colormap name
    #     show_mask : bool
    #         If True, overlay the mask on the heatmap
    #     """
    #     fig, ax = plt.subplots(figsize=figsize)
        
    #     # Create custom colormap with grey for masked values
    #     if show_mask:
    #         # Create a copy of the data
    #         plot_data = self.masked_df.copy()
            
    #         # Create custom colormap
    #         cmap_obj = plt.cm.get_cmap(cmap)
    #         cmap_obj.set_bad(color='lightgray')
            
    #         im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap_obj, 
    #                       interpolation='nearest', extent=[0, len(self.df.index), 0, len(self.df.columns)])
    #     else:
    #         im = ax.imshow(self.masked_df.T, aspect='auto', cmap=cmap, 
    #                       interpolation='nearest', extent=[0, len(self.df.index), 0, len(self.df.columns)])
        
    #     # Set labels
    #     ax.set_xlabel('Time Index')
    #     ax.set_ylabel('Node')
    #     ax.set_title(f'{self.var_name} - Spatiotemporal Signal Heatmap')
        
    #     # Add colorbar
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     cbar.set_label('Signal Value')
        
    #     # Set tick labels
    #     if len(self.df.index) < 50:
    #         ax.set_xticks(range(0, len(self.df.index), max(1, len(self.df.index)//10)))
    #         ax.set_xticklabels([str(idx)[:10] for idx in self.df.index[::max(1, len(self.df.index)//10)]], 
    #                            rotation=45, ha='right')
        
    #     # Add year lines if datetime index
    #     try:
    #         time_index = pd.to_datetime(self.df.index)
    #         self._add_year_lines_heatmap(ax, time_index)
    #     except:
    #         pass
        
    #     plt.tight_layout()
    #     return fig, ax
    
    def plot_heatmap(self, figsize=(12, 8), cmap='viridis', show_mask=False):
        """
        Heat-map of the spatio-temporal signal.
        Missing dates (outside Apr–Oct) are rendered in grey.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # ------------------------------------------------------------------
        # 1. Build a *full-year* index at the native frequency of the data
        # ------------------------------------------------------------------
        time_index = pd.to_datetime(self.df.index)
        freq = pd.infer_freq(time_index) or 'D'                   # fall back to daily
        full_index = pd.date_range(start=time_index.min().replace(month=1, day=1),
                                end  =time_index.max().replace(month=12, day=31),
                                freq =freq)

        # Re-index the masked data; new rows are NaN  → plotted as “bad”
        plot_df = self.masked_df.reindex(full_index)

        # ------------------------------------------------------------------
        # 2. Configure colormap so NaN → light grey
        # ------------------------------------------------------------------
        cmap_obj = plt.cm.get_cmap(cmap).copy()                   # avoid mutating global cmap
        cmap_obj.set_bad(color='lightgray')

        # ------------------------------------------------------------------
        # 3. Draw the heat-map
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_df.T, aspect='auto', cmap=cmap_obj,
                    interpolation='nearest',
                    extent=[0, len(full_index), 0, len(self.df.columns)])

        # Axis labels and title
        ax.set_xlabel('Time index')
        ax.set_ylabel('Node')
        ax.set_title(f'{self.var_name} – spatiotemporal signal')

        # Colour-bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Signal value')

        # X-ticks: show ≤10 evenly spaced labels
        step = max(1, len(full_index) // 10)
        ax.set_xticks(range(0, len(full_index), step))
        ax.set_xticklabels([idx.strftime('%d-%b') for idx in full_index[::step]],
                        rotation=45, ha='right')

        # Optional vertical year lines
        self._add_year_lines_heatmap(ax, full_index)

        plt.tight_layout()
        return fig, ax



    def plot_node_traces(self, nodes=None, figsize=(12, 6), alpha=0.7):
        """
        Plot time series for specific nodes.
        
        Parameters:
        -----------
        nodes : list or None
            List of node names to plot. If None, plot first 5 nodes
        figsize : tuple
            Figure size
        alpha : float
            Transparency for lines
        """
        if nodes is None:
            nodes = self.df.columns[:min(5, len(self.df.columns))]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if index can be converted to datetime
        try:
            time_index = pd.to_datetime(self.df.index)
            use_datetime = True
        except:
            time_index = self.df.index
            use_datetime = False
        
        for node in nodes:
            # Plot only valid data points
            valid_data = self.masked_df[node]
            if use_datetime:
                # Create a temporary series with datetime index
                temp_series = pd.Series(valid_data.values, index=time_index)
                ax.plot(temp_series.index, temp_series.values, label=f'Node {node}', 
                       alpha=alpha, marker='o', markersize=3)
            else:
                ax.plot(range(len(valid_data)), valid_data.values, label=f'Node {node}', 
                       alpha=alpha, marker='o', markersize=3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Value')
        ax.set_title(f'{self.var_name} - Node Signal Traces')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format time axis and add year lines
        if use_datetime:
            self._format_time_axis(ax)
            self._add_year_lines(ax)
        else:
            # For non-datetime index, show subset of labels
            n_labels = min(10, len(self.df.index))
            indices = np.linspace(0, len(self.df.index)-1, n_labels, dtype=int)
            ax.set_xticks(indices)
            ax.set_xticklabels([str(self.df.index[i])[:10] for i in indices], 
                             rotation=45, ha='right')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_mask_coverage(self, figsize=(12, 6)):
        """
        Visualize the data coverage (mask) over time and nodes.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Mask heatmap
        im = ax1.imshow(self.mask.T, aspect='auto', cmap='RdYlGn', 
                       interpolation='nearest', vmin=0, vmax=1,
                       extent=[0, len(self.df.index), 0, len(self.df.columns)])
        ax1.set_ylabel('Node')
        ax1.set_title(f'{self.var_name} - Data Availability Mask (Green=Valid, Red=Invalid)')
        
        # Format x-axis for heatmap
        time_indices = np.linspace(0, len(self.df.index)-1, min(10, len(self.df.index)), dtype=int)
        ax1.set_xticks(time_indices)
        ax1.set_xticklabels([str(self.df.index[i])[:10] for i in time_indices], 
                           rotation=45, ha='right')
        
        # Add year lines to mask heatmap if datetime index
        try:
            time_index = pd.to_datetime(self.df.index)
            self._add_year_lines_heatmap(ax1, time_index)
        except:
            pass
        
        # Coverage over time
        coverage = self.mask.sum(axis=1) / len(self.mask.columns) * 100
        
        # Check if index can be converted to datetime
        try:
            time_index = pd.to_datetime(self.df.index)
            use_datetime = True
        except:
            time_index = self.df.index
            use_datetime = False
        
        if use_datetime:
            # Create a temporary series with datetime index
            coverage_series = pd.Series(coverage.values, index=time_index)
            ax2.plot(coverage_series.index, coverage_series.values, color='darkgreen', linewidth=2)
            ax2.fill_between(coverage_series.index, coverage_series.values, alpha=0.3, color='green')
        else:
            ax2.plot(range(len(coverage)), coverage.values, color='darkgreen', linewidth=2)
            ax2.fill_between(range(len(coverage)), coverage.values, alpha=0.3, color='green')
            
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coverage %')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        
        # Format time axis and add year lines for coverage plot
        if use_datetime:
            self._format_time_axis(ax2)
            self._add_year_lines(ax2)
        else:
            # For non-datetime index, show subset of labels
            n_labels = min(10, len(self.df.index))
            indices = np.linspace(0, len(self.df.index)-1, n_labels, dtype=int)
            ax2.set_xticks(indices)
            ax2.set_xticklabels([str(self.df.index[i])[:10] for i in indices], 
                             rotation=45, ha='right')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_statistics(self, figsize=(15, 10)):
        """
        Plot statistical summary of the data.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Distribution of values (only valid data)
        ax = axes[0, 0]
        valid_values = self.masked_df.values.flatten()
        valid_values = valid_values[~np.isnan(valid_values)]
        ax.hist(valid_values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Signal Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.var_name} - Distribution of Valid Signal Values')
        
        # 2. Node-wise statistics
        ax = axes[0, 1]
        node_means = self.masked_df.mean()
        node_stds = self.masked_df.std()
        x = range(len(node_means))
        ax.errorbar(x, node_means, yerr=node_stds, fmt='o', capsize=5, alpha=0.7)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Mean ± Std')
        ax.set_title(f'{self.var_name} - Node-wise Statistics')
        ax.grid(True, alpha=0.3)
        
        # 3. Temporal statistics
        ax = axes[1, 0]
        temporal_means = self.masked_df.mean(axis=1)
        temporal_stds = self.masked_df.std(axis=1)
        
        # Check if index can be converted to datetime
        try:
            time_index = pd.to_datetime(self.df.index)
            use_datetime = True
        except:
            time_index = self.df.index
            use_datetime = False
            
        if use_datetime:
            # Create temporary series with datetime index
            mean_series = pd.Series(temporal_means.values, index=time_index)
            std_series = pd.Series(temporal_stds.values, index=time_index)
            ax.plot(mean_series.index, mean_series.values, label='Mean', linewidth=2)
            ax.fill_between(mean_series.index, 
                           mean_series.values - std_series.values, 
                           mean_series.values + std_series.values, 
                           alpha=0.3, label='±1 Std')
        else:
            ax.plot(range(len(temporal_means)), temporal_means.values, label='Mean', linewidth=2)
            ax.fill_between(range(len(temporal_means)), 
                           temporal_means.values - temporal_stds.values, 
                           temporal_means.values + temporal_stds.values, 
                           alpha=0.3, label='±1 Std')
            
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Value')
        ax.set_title(f'{self.var_name} - Temporal Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format time axis and add year lines
        if use_datetime:
            self._format_time_axis(ax)
            self._add_year_lines(ax)
        else:
            # For non-datetime index, show subset of labels
            n_labels = min(10, len(self.df.index))
            indices = np.linspace(0, len(self.df.index)-1, n_labels, dtype=int)
            ax.set_xticks(indices)
            ax.set_xticklabels([str(self.df.index[i])[:10] for i in indices], 
                             rotation=45, ha='right')
        
        # 4. Correlation matrix (only for valid data pairs)
        ax = axes[1, 1]
        corr_matrix = self.masked_df.corr()
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')
        ax.set_title(f'{self.var_name} - Node Correlation Matrix')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout()
        return fig, axes
    
    def create_animation(self, window_size=10, interval=100, figsize=(10, 6)):
        """
        Create an animation showing the signal evolution over time.
        
        Parameters:
        -----------
        window_size : int
            Number of time steps to show in each frame
        interval : int
            Delay between frames in milliseconds
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize plot
        line_objects = []
        for i, node in enumerate(self.df.columns[:5]):  # Animate first 5 nodes
            line, = ax.plot([], [], label=f'Node {node}', alpha=0.7)
            line_objects.append(line)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Value')
        ax.set_title(f'{self.var_name} - Signal Animation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def init():
            for line in line_objects:
                line.set_data([], [])
            return line_objects
        
        def animate(frame):
            start_idx = max(0, frame - window_size)
            end_idx = frame + 1
            
            time_window = self.df.index[start_idx:end_idx]
            
            for i, (line, node) in enumerate(zip(line_objects, self.df.columns[:5])):
                valid_data = self.masked_df[node].iloc[start_idx:end_idx]
                line.set_data(time_window, valid_data)
            
            # Update axis limits
            if len(time_window) > 0:
                ax.set_xlim(time_window[0], time_window[-1])
                all_valid = self.masked_df.iloc[start_idx:end_idx, :5].values.flatten()
                all_valid = all_valid[~np.isnan(all_valid)]
                if len(all_valid) > 0:
                    ax.set_ylim(np.min(all_valid) * 0.9, np.max(all_valid) * 1.1)
            
            return line_objects
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.df), interval=interval, 
                           blit=True, repeat=True)
        
        return fig, anim

    def save_all_plots(self):
        """
        Generate and save all visualization plots.
        """
        print(f"Saving all plots to '{self.output_dir}/' directory...")
        
        # 1. Heatmap
        fig1, _ = self.plot_heatmap(show_mask=True)
        plt.savefig(os.path.join(self.output_dir, 'spatiotemporal_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("✓ Heatmap saved")
        
        # 2. Node traces
        fig2, _ = self.plot_node_traces()
        plt.savefig(os.path.join(self.output_dir, 'node_traces.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("✓ Node traces saved")
        
        # 3. Mask coverage
        fig3, _ = self.plot_mask_coverage()
        plt.savefig(os.path.join(self.output_dir, 'mask_coverage.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("✓ Mask coverage saved")
        
        # 4. Statistics
        fig4, _ = self.plot_statistics()
        plt.savefig(os.path.join(self.output_dir, 'statistics_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("✓ Statistics summary saved")
        
        print(f"\nAll plots saved to '{self.output_dir}/' directory!")


# Example usage
def demo_visualization():
    """
    Demonstrate the visualization capabilities with synthetic data.
    """
    # Generate synthetic data spanning multiple years
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=1000, freq='D')  # ~2.7 years of daily data
    nodes = [f'Node_{i}' for i in range(20)]
    
    # Create synthetic signals
    data = np.zeros((len(dates), len(nodes)))
    for i in range(len(nodes)):
        # Different patterns for different nodes
        if i < 5:
            data[:, i] = np.sin(np.linspace(0, 4*np.pi, len(dates)) + i) + np.random.normal(0, 0.1, len(dates))
        elif i < 10:
            data[:, i] = np.cos(np.linspace(0, 6*np.pi, len(dates)) - i) + np.random.normal(0, 0.15, len(dates))
        else:
            data[:, i] = np.random.normal(0, 1, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates, columns=nodes)
    
    # Create mask (simulate nodes turning on/off)
    mask = np.ones_like(data)
    # Make some nodes inactive at certain times
    mask[20:40, 5:10] = 0
    mask[60:80, 15:20] = 0
    mask[0:30, 18] = 0
    
    # Add some random dropouts
    random_mask = np.random.random(mask.shape) > 0.1
    mask = mask * random_mask
    
    # Create visualizer with variable name
    viz = SpatiotemporalVisualizer(df, mask, var_name='Temperature')
    
    # Create plots folder if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create all visualizations
    print("Creating visualizations...")
    
    # 1. Heatmap
    fig1, _ = viz.plot_heatmap(show_mask=True)
    plt.savefig('plots/spatiotemporal_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Heatmap saved")
    
    # 2. Node traces
    fig2, _ = viz.plot_node_traces()
    plt.savefig('plots/node_traces.png', dpi=300, bbox_inches='tight')
    print("✓ Node traces saved")
    
    # 3. Mask coverage
    fig3, _ = viz.plot_mask_coverage()
    plt.savefig('plots/mask_coverage.png', dpi=300, bbox_inches='tight')
    print("✓ Mask coverage saved")
    
    # 4. Statistics
    fig4, _ = viz.plot_statistics()
    plt.savefig('plots/statistics_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Statistics summary saved")
    
    # 5. Animation (save as GIF)
    fig5, anim = viz.create_animation()
    anim.save('plots/signal_animation.gif', writer='pillow')
    plt.close(fig5)
    print("✓ Animation saved")
    
    print(f"\nAll visualizations saved to 'plots/' directory!")
    
    return viz


if __name__ == "__main__":
    
    # nb_ostrinia, incrementing_ostrinia
    target = 'incrementing_ostrinia'

    # Example of using with your own data:
    dataset = Ostrinia(root="datasets", target=target, smooth=True)

    df = dataset.target
    mask = dataset.mask[:, :, 0]  # Transpose to match the shape of df
    
    # If your index is not datetime, convert it:
    df.index = pd.to_datetime(df.index)

    # Create visualizer with variable name
    viz = SpatiotemporalVisualizer(df, mask, var_name=target)
    viz.save_all_plots()

