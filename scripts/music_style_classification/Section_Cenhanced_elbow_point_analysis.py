import numpy as np
import pandas as pd
import sqlite3
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Configure font display (support Unicode, ensure minus sign displays correctly)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use DejaVu Sans as sans-serif font
plt.rcParams['axes.unicode_minus'] = False         # Ensure minus signs are displayed correctly

class EnhancedElbowAnalyzer:
    def __init__(self, db_path=None, table_name="KmeanSample", features=None,
                 k_range=range(4, 16), sample_size=8693,  # consistent with figure
                 n_init=15,               # more stable
                 init="k-means++",        # explicit
                 n_seeds=5,               # multiple random seeds for each K
                 best_k=None,
                 scale=True,              # whether to use StandardScaler
                 random_state=42,
                 plot_dpi=300, figsize=(12, 5),
                 knee_method="kneedle"):  # or "diff"
        """
        Enhanced Elbow Point analyzer
        
        Parameters:
        - db_path: path to the database
        - table_name: table name
        - features: list of feature column names
        - k_range: range of K values
        - sample_size: sample size
        - n_init: number of KMeans initializations
        - best_k: manually specified best K (if None, automatically detected)
        - plot_dpi: figure DPI
        - figsize: figure size
        """
        self.db_path = "SPOTIFY_ANALYSIS_PROJECT/data/task1/spotify_database.db"
        self.table_name = table_name
        self.features = features or ['Danceability', 'Loudness', 'Speechiness',
                                     'Acousticness', 'Instrumentalness', 'Valence']
        self.k_range = k_range
        self.sample_size = sample_size
        self.n_init = n_init
        self.user_best_k = best_k
        self.plot_dpi = plot_dpi
        self.figsize = figsize
        
        # Color settings
        self.main_color = "#1f77b4"          # main color
        self.highlight_color = "#d62728"     # highlight color (red)
        self.grid_color = "gray"             # grid color
        self.text_color = "#333333"          # text color
        self.samples_note_color = "#666666"  # color for sample-size annotation
        
        # Initialize data and result storage
        self.X = None
        self.X_sample = None
        self.inertias = []
        self.silhouettes = []
        self.best_k = 8
        self.elbow_k = None
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data"""
        print("Connecting to database and loading data...")
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM {self.table_name};", conn)
        
        # Extract features
        self.X = df[self.features]
        print(f"Data loaded, total {len(self.X)} records")
        
        # Sample data
        sample_size = min(self.sample_size, len(self.X))
        sample_idx = np.random.choice(range(len(self.X)), size=sample_size, replace=False)
        self.X_sample = self.X.iloc[sample_idx]
        print(f"Sampled {sample_size} records for analysis")
    
    def detect_elbow(self, inertias, k_range):
        """
        Detect elbow point using least squares method.
        Idea: find the point with the largest deviation from the line between
        the first and last points.
        """
        # Convert to numpy arrays
        inertias_np = np.array(inertias)
        k_np = np.array(list(k_range))
        
        # Compute the line from the first to the last point
        x1, y1 = k_np[0], inertias_np[0]
        x2, y2 = k_np[-1], inertias_np[-1]
        
        # Line equation: y = m x + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # Compute perpendicular distance from each point to the line
        distances = np.abs(m * k_np - inertias_np + b) / np.sqrt(m**2 + 1)
        
        # Elbow point = point with maximum distance
        elbow_idx = np.argmax(distances)
        elbow_k = k_np[elbow_idx]
        
        return elbow_k, elbow_idx
    
    def run_analysis(self):
        """Run KMeans for each K and compute inertia and silhouette score"""
        print(f"Starting analysis for K in range: {min(self.k_range)} to {max(self.k_range)}")
        
        # Standardize data (important for KMeans)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_sample)
        
        for k in self.k_range:
            print(f"Computing K={k}...")
            model = KMeans(n_clusters=k, init='k-means++', n_init=self.n_init,
                          max_iter=200, tol=0.001, random_state=42)
            
            labels = model.fit_predict(X_scaled)
            self.inertias.append(model.inertia_)
            
            # Compute silhouette score
            silhouette_avg = silhouette_score(X_scaled, labels)
            self.silhouettes.append(silhouette_avg)
            print(f"  K={k} silhouette score: {silhouette_avg:.4f}")
        
        # Detect elbow point
        self.elbow_k, self.elbow_idx = self.detect_elbow(self.inertias, self.k_range)
        
        # Determine best K
        if self.user_best_k is not None:
            self.best_k = self.user_best_k
            print(f"Using user-specified best K: {self.best_k}")
        else:
            # Use K with maximum silhouette score
            max_silhouette_idx = np.argmax(self.silhouettes)
            self.best_k = list(self.k_range)[max_silhouette_idx]
            print(f"Automatically detected best K (based on silhouette score): {self.best_k}")
        
        print(f"Elbow detection result: K={self.elbow_k}")
    
    def plot_enhanced_elbow(self, output_dir=None):
        """Plot the enhanced elbow and silhouette analysis figure"""
        if len(self.inertias) == 0 or len(self.silhouettes) == 0:
            raise ValueError("Please run run_analysis() first")
        
        # Set figure style
        plt.style.use('seaborn-v0_8-white')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.plot_dpi)
        k_values = list(self.k_range)
        
        # ===== Left: Elbow Method plot =====
        ax1.plot(
            k_values, self.inertias,
            marker='o', linewidth=2.0, markersize=7,
            color=self.main_color, linestyle='-', markerfacecolor=self.main_color
        )
        
        # Highlight elbow point
        elbow_idx = k_values.index(self.elbow_k)
        ax1.scatter(
            self.elbow_k, self.inertias[elbow_idx],
            s=100, color=self.highlight_color,
            edgecolor='black', linewidth=1.5, zorder=5
        )
        ax1.text(
            self.elbow_k + 0.2, self.inertias[elbow_idx], 'Elbow point',
            fontsize=9, color=self.text_color, va='bottom'
        )
        
        # Draw vertical line for best K
        best_k_idx = k_values.index(self.best_k)
        ax1.axvline(
            x=self.best_k, color=self.highlight_color,
            linestyle='--', linewidth=1.5, alpha=0.8, zorder=3
        )
        ax1.text(
            self.best_k,
            ax1.get_ylim()[0] + 0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
            f'K = {self.best_k}',
            fontsize=10, color=self.highlight_color,
            va='bottom', rotation=90
        )
        
        # Titles and labels
        ax1.set_title('Elbow Method for Optimal K', fontsize=21, fontweight='bold', color=self.text_color)
        ax1.set_xlabel('Number of clusters (K)', fontsize=18, color=self.text_color)
        ax1.set_ylabel('Inertia (Within-cluster SSE)', fontsize=18, color=self.text_color)
        
        # Grid
        ax1.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
        
        # Tick label size
        ax1.tick_params(axis='both', labelsize=10, color=self.text_color)
        
        # ===== Right: Silhouette Score plot =====
        ax2.plot(
            k_values, self.silhouettes,
            marker='o', linewidth=2.0, markersize=7,
            color=self.main_color, linestyle='-', markerfacecolor=self.main_color
        )
        
        # Highlight maximum silhouette score
        max_silhouette_idx = np.argmax(self.silhouettes)
        max_k = k_values[max_silhouette_idx]
        max_silhouette = self.silhouettes[max_silhouette_idx]
        
        ax2.scatter(
            max_k, max_silhouette,
            s=100, color=self.highlight_color,
            edgecolor='black', linewidth=1.5, zorder=5
        )
        ax2.text(
            max_k + 0.2, max_silhouette, 'Max silhouette',
            fontsize=9, color=self.text_color, va='bottom'
        )
        
        # Vertical line for best K
        ax2.axvline(
            x=self.best_k, color=self.highlight_color,
            linestyle='--', linewidth=1.5, alpha=0.8, zorder=3
        )
        ax2.text(
            self.best_k,
            ax2.get_ylim()[0] + 0.03 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
            f'K = {self.best_k}',
            fontsize=10, color=self.highlight_color,
            va='bottom', rotation=90
        )
        
        # Titles and labels
        ax2.set_title('Silhouette Score across K', fontsize=21, fontweight='bold', color=self.text_color)
        ax2.set_xlabel('Number of clusters (K)', fontsize=18, color=self.text_color)
        ax2.set_ylabel('Average Silhouette Score', fontsize=18, color=self.text_color)
        
        # Grid
        ax2.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)
        
        # Tick label size
        ax2.tick_params(axis='both', labelsize=10, color=self.text_color)
        
        # Ensure same x-range on both plots
        min_k, max_k = min(k_values), max(k_values)
        ax1.set_xlim(min_k - 0.5, max_k + 0.5)
        ax2.set_xlim(min_k - 0.5, max_k + 0.5)
        
        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.95, top=1.0, bottom=0.98, wspace=0.3)
        # Shift main plots upwards to reserve more space at the bottom (for a clear footer)
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.33, wspace=0.3)  # UPDATED

        # Add a "footer" axes at the bottom ([left, bottom, width, height] in figure coordinates)
        footer_ax = fig.add_axes([0.08, 0.03, 0.87, 0.22])  # NEW: occupy about 22% of the bottom height
        footer_ax.axis('off')

        # Put summary text in the middle of the footer (can be moved lower by adjusting parameters)
        summary_text = (
            f"The final choice of K = {self.best_k} reflects a balance between "
            f"ARI performance and variance, not just the Elbow and Silhouette metrics."
        )
        footer_ax.text(
            0.5, 0.6, summary_text,
            fontsize=15, color=self.text_color,
            ha='center', va='center', wrap=True
        )  # NEW

        # Right-side vertical n_samples text (placed along the right edge, vertically centered)
        fig.text(
            0.985, 0.5, f'n_samples={len(self.X_sample)}',
            rotation=270, rotation_mode='anchor',
            fontsize=9, color=self.samples_note_color,
            ha='right', va='center'
        )
        
        # Save figure
        output_dir = output_dir or 'SPOTIFY_ANALYSIS_PROJECT/scripts/music_style_classification/results'
        svg_filename = os.path.join(output_dir, 'enhanced_elbow_analysis.svg')
        png_filename = os.path.join(output_dir, 'enhanced_elbow_analysis.png')
        
        plt.savefig(svg_filename, format='svg', dpi=self.plot_dpi, bbox_inches='tight')
        plt.savefig(png_filename, format='png', dpi=self.plot_dpi, bbox_inches='tight')
        
        print(f"Saved enhanced elbow analysis high-resolution vector figure to: {svg_filename}")
        print(f"Saved enhanced elbow analysis high-resolution PNG figure to: {png_filename}")
        
        plt.show()
        
        return fig, (ax1, ax2)
    
    def get_results_summary(self):
        """Get summary of analysis results"""
        if len(self.inertias) == 0 or len(self.silhouettes) == 0:
            raise ValueError("Please run run_analysis() first")
        
        summary = {
            "best_k": self.best_k,
            "elbow_k": self.elbow_k,
            "k_range": self.k_range,
            "inertias": dict(zip(self.k_range, self.inertias)),
            "silhouettes": dict(zip(self.k_range, self.silhouettes)),
            "best_k_silhouette": self.silhouettes[list(self.k_range).index(self.best_k)],
            "sample_size": len(self.X_sample)
        }
        
        print("\n===== Analysis Summary =====")
        print(f"Recommended best K: {summary['best_k']}")
        print(f"Elbow detection result: {summary['elbow_k']}")
        print(f"Silhouette score at best K: {summary['best_k_silhouette']:.4f}")
        print(f"Sample size used for analysis: {summary['sample_size']}")
        
        return summary

# Main function
def main():
    print("========================================")
    print("        Enhanced Elbow Point Analysis Tool")
    print("========================================")
    print("Based on stability analysis, we fix K=10 as the preferred number of clusters")
    print("========================================")
    
    # Create analyzer instance
    analyzer = EnhancedElbowAnalyzer(
        k_range=range(4, 16),  # Range of K values
        sample_size=10000,     # Sample size
        n_init=10,             # Number of KMeans initializations
        best_k=8,              # Fix K=8 as the chosen optimal K
        figsize=(12, 5)        # Figure size
    )
    
    # Run analysis
    analyzer.run_analysis()
    
    # Plot enhanced figure
    analyzer.plot_enhanced_elbow()
    
    # Get summary of results
    analyzer.get_results_summary()
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()

