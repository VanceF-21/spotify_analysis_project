import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import itertools
import time

class KMeansStabilityAnalyzer:
    def __init__(self, db_path=None, table_name="KmeanSample", features=None,
                 k_values=[4, 7, 8, 10], n_seeds=100, sample_size=10000,
                 n_init=10, output_dir="results"):
        """
        K-means clustering stability analyzer
        
        Parameters:
        - db_path: path to the SQLite database
        - table_name: table name in the database
        - features: list of feature column names
        - k_values: list of K values to analyze
        - n_seeds: number of random seeds (runs) for each K value
        - sample_size: sample size used for stability analysis
        """
        self.db_path = db_path or r"SPOTIFY_ANALYSIS_PROJECT/data/task1/spotify_database.db"
        self.table_name = table_name
        self.features = features or ['Danceability', 'Loudness', 'Speechiness',
                                     'Acousticness', 'Instrumentalness', 'Valence']
        self.k_values = k_values
        self.n_seeds = n_seeds
        self.sample_size = sample_size
        self.n_init = n_init
        self.output_dir = output_dir
        
        # Initialize data and result containers
        self.X = None
        self.X_sample = None
        self.results = {}
        self.run_results_df = None
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data"""
        print(f"Connecting to database and loading data...")
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM {self.table_name};", conn)
        
        # Extract features
        self.X = df[self.features]
        print(f"Data loaded, total {len(self.X)} records")
        
        # Sample data
        sample_size = min(self.sample_size, len(self.X))
        sample_idx = np.random.choice(range(len(self.X)), size=sample_size, replace=False)
        self.X_sample = self.X.iloc[sample_idx]
        print(f"Sampled {sample_size} records for stability analysis")
        
        # Standardize data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_sample)
    
    def run_stability_analysis(self):
        """
        Run stability analysis: for each K value, use multiple random seeds
        """
        print(f"\nStarting stability analysis, K values: {self.k_values}")
        print(f"Running {self.n_seeds} random seeds for each K value")
        print(f"Using K-means++ initialization, n_init={self.n_init}")
        
        # Create a DataFrame to store all run results
        run_results_data = []
        
        for k in self.k_values:
            print(f"\nAnalyzing K={k} ...")
            
            # Containers for current K's results
            silhouettes = []
            cluster_labels = []
            inertias = []
            seeds = []
            
            # Run multiple random seeds
            for seed_idx in range(self.n_seeds):
                start_time = time.time()
                seed = seed_idx  # use index as random seed
                
                # Create KMeans model with current seed
                model = KMeans(n_clusters=k, init='k-means++', n_init=self.n_init,
                              max_iter=200, tol=0.001, random_state=seed)
                
                # Fit model and get labels
                labels = model.fit_predict(self.X_scaled)
                
                # Compute metrics
                silhouette = silhouette_score(self.X_scaled, labels)
                silhouettes.append(silhouette)
                cluster_labels.append(labels)
                inertias.append(model.inertia_)
                seeds.append(seed)
                
                end_time = time.time()
                print(f"  Run {seed_idx+1}/{self.n_seeds}: Silhouette score = {silhouette:.4f}, Time = {end_time-start_time:.2f}s")
                
                # Temporarily store current run result
                run_results_data.append({
                    'K': k,
                    'run_id': seed_idx,
                    'random_state': seed,
                    'silhouette_mean': silhouette,
                    'inertia': model.inertia_,
                    'ari_run': None  # will be filled later
                })
            
            # Find median run (the run whose silhouette score is closest to the median)
            median_silhouette = np.median(silhouettes)
            median_run_idx = np.argmin(np.abs(np.array(silhouettes) - median_silhouette))
            median_labels = cluster_labels[median_run_idx]
            print(f"  Median run: run_id={median_run_idx}, Silhouette score={silhouettes[median_run_idx]:.4f}")
            
            # Compute ARI between each run and the median run
            for seed_idx in range(self.n_seeds):
                run_idx_in_data = len(run_results_data) - self.n_seeds + seed_idx
                run_results_data[run_idx_in_data]['ari_run'] = adjusted_rand_score(
                    cluster_labels[seed_idx], median_labels
                )
            
            # Compute intra-group stability (average ARI between different seeds)
            ari_scores = []
            for i, j in itertools.combinations(range(self.n_seeds), 2):
                ari = adjusted_rand_score(cluster_labels[i], cluster_labels[j])
                ari_scores.append(ari)
            
            avg_ari = np.mean(ari_scores)
            
            # Store results
            self.results[k] = {
                'silhouettes': silhouettes,
                'silhouette_mean': np.mean(silhouettes),
                'silhouette_std': np.std(silhouettes),
                'inertias': inertias,
                'ari_scores': ari_scores,
                'avg_ari': avg_ari,
                'cluster_labels': cluster_labels,
                'median_run_idx': median_run_idx
            }
            
            print(f"  K={k} results:")
            print(f"  - Silhouette score: {np.mean(silhouettes):.4f} ± {np.std(silhouettes):.4f}")
            print(f"  - Intra-group stability (Average ARI): {avg_ari:.4f}")
        
        # Create full results DataFrame
        self.run_results_df = pd.DataFrame(run_results_data)
        print(f"\nStability analysis completed! A total of {len(self.run_results_df)} runs were recorded")
    
    def print_results_table(self):
        """
        Print results in table format
        """
        if not self.results:
            raise ValueError("Please run run_stability_analysis() first")
        
        print("\n===== K-means Clustering Stability Analysis Results =====")
        print("K  | Silhouette (mean ± std)   | Intra-group Stability (Mean ARI)")
        print("----|---------------------------|-------------------------------")
        
        for k in sorted(self.k_values):
            res = self.results[k]
            sil_mean = res['silhouette_mean']
            sil_std = res['silhouette_std']
            avg_ari = res['avg_ari']
            
            print(f"{k:2d}  | {sil_mean:.4f} ± {sil_std:.4f}       | {avg_ari:.4f}")
    
    def get_results_summary(self):
        """
        Get summary of results
        """
        if not self.results:
            raise ValueError("Please run run_stability_analysis() first")
        
        summary = {}
        for k in self.k_values:
            res = self.results[k]
            summary[k] = {
                'silhouette_mean': res['silhouette_mean'],
                'silhouette_std': res['silhouette_std'],
                'avg_ari': res['avg_ari']
            }
        
        return summary
    
    def save_results_to_csv(self, filename="cluster_stability_runs.csv"):
        """
        Save run results to CSV file
        """
        if self.run_results_df is None:
            raise ValueError("Please run run_stability_analysis() first")
        
        filepath = os.path.join(self.output_dir, filename)
        self.run_results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def generate_boxplot(self, filename="ari_boxplot_k4710.png"):
        """
        Generate ARI boxplot
        """
        if self.run_results_df is None:
            raise ValueError("Please run run_stability_analysis() first")
        
        plt.figure(figsize=(10, 6))
        
        # Keep default font settings to support English display
        plt.rcParams['axes.unicode_minus'] = False    # Ensure minus sign is displayed correctly
        
        # Create boxplot
        sns.boxplot(x='K', y='ari_run', data=self.run_results_df)
        
        # Add title and labels
        plt.title(f'K-means++ Clustering Stability Analysis (N={self.n_seeds}, n_init={self.n_init})', fontsize=16)
        plt.xlabel('Number of Clusters K', fontsize=12)
        plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
        
        # Add gridlines
        plt.grid(axis='y', alpha=0.3)
        
        # Add annotation for number of runs per K
        for k in self.k_values:
            count = len(self.run_results_df[self.run_results_df['K'] == k])
            plt.annotate(
                f'n={count}', 
                xy=(self.k_values.index(k), self.run_results_df[self.run_results_df['K'] == k]['ari_run'].min() - 0.02),
                ha='center', fontsize=10
            )
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot saved to: {filepath}")
        return filepath
    
    def generate_violinplot(self, filename="ari_violinplot_k47810.png"):
        """
        Generate ARI violin plot with customized style
        """
        if self.run_results_df is None:
            raise ValueError("Please run run_stability_analysis() first")
        
        plt.figure(figsize=(10, 6))
        
        # Get current axis
        ax = plt.gca()
        
        # Set background to white and turn off grid
        ax.set_facecolor('white')
        ax.grid(False)
        
        # Keep default font settings to support English display
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create customized violin plot
        violin_plot = sns.violinplot(
            x='K', 
            y='ari_run', 
            data=self.run_results_df, 
            inner=None,      # Do not use default inner box
            color='#ff66cc', # Bright pink fill
            linewidth=0.8,   # Thin border line
            alpha=0.8,       # Transparency
            ax=ax
        )
        
        # Add black boxplot (width=0.1)
        box_plot = sns.boxplot(
            x='K', 
            y='ari_run', 
            data=self.run_results_df, 
            width=0.1,  # Box width
            boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.8},
            whiskerprops={'color': 'black', 'linewidth': 0.8},
            capprops={'color': 'black', 'linewidth': 0.8},
            ax=ax
        )
        
        # Compute and plot white solid dots as medians
        for k in self.k_values:
            # Get data for current K
            data = self.run_results_df[self.run_results_df['K'] == k]['ari_run']
            # Compute quartiles
            q1 = np.percentile(data, 25)
            median = np.median(data)
            q3 = np.percentile(data, 75)
            
            # Find x-position of current K on axis
            k_idx = self.k_values.index(k)
            
            # Plot white solid dot as median
            plt.scatter(
                k_idx, 
                median, 
                s=5,           # Point size
                color='white', 
                edgecolor='black',  # Black border
                linewidth=0.5, 
                zorder=5       # Ensure point is on top
            )
        
        # Add title and labels (English)
        plt.title(f'K-means++ Clustering Stability Analysis (N={self.n_seeds}, n_init={self.n_init})', 
                 fontsize=16, color='black')
        plt.xlabel('Number of Clusters K', fontsize=12, color='black')
        plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12, color='black')
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved to: {filepath}")
        return filepath
    
    def analyze_and_visualize(self):
        """
        Run the full analysis and visualization workflow
        """
        # Run stability analysis
        self.run_stability_analysis()
        
        # Print results table
        self.print_results_table()
        
        # Save results to CSV
        csv_path = self.save_results_to_csv()
        
        # Generate boxplot
        boxplot_path = self.generate_boxplot()
        
        # Generate violin plot
        violinplot_path = self.generate_violinplot()
        
        print(f"\nComplete analysis workflow finished!")
        print(f"- Result data: {csv_path}")
        print(f"- Boxplot: {boxplot_path}")
        print(f"- Violin plot: {violinplot_path}")

# Main function
def main():
    print("========================================")
    print("        K-means Clustering Stability Analysis Tool")
    print("========================================")
    print("Analysis: Run multiple random seeds for specified K values, calculate ARI stability metrics")
    print("Goal: Intuitively show stability differences between runs for different K values")
    print("========================================")
    
    # Create analyzer instance
    analyzer = KMeansStabilityAnalyzer(
        k_values=[4, 7, 8, 10],  # K values to analyze
        n_seeds=50,              # Run 50 times for each K value
        n_init=50,               # Number of initial center attempts
        sample_size=10000        # Sample size
    )
    
    # Run full analysis and visualization workflow
    analyzer.analyze_and_visualize()
    
    print("\n========================================")
    print("Analysis Description:")
    print(f"1. Each K value was run {analyzer.n_seeds} times with different random seeds")
    print(f"2. Using K-means++ initialization, n_init={analyzer.n_init}")
    print(f"3. ARI calculated based on comparison with median run")
    print(f"4. Data sample size: {analyzer.sample_size}")
    print(f"5. All result files are saved in the results directory")
    print("========================================")

if __name__ == "__main__":
    main()

