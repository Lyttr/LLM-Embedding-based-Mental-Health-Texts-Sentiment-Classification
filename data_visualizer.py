import logging
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIG_SIZE, PLOT_STYLE, COLOR_PALETTE

class DataVisualizer:
    def __init__(self, save_dir='plots/data_analysis'):
        plt.style.use(PLOT_STYLE)
        self.fig_size = FIG_SIZE
        self.colors = sns.color_palette(COLOR_PALETTE)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _save_plot(self, filename):
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_data(self, df):
        try:
            df['text_length'] = df['statement'].str.len()
            logging.info("Generating data visualizations...")
            
            self.plot_len_dist(df)
            self.plot_cls_dist(df)
            self.plot_len_by_cls(df)
            
        except Exception as e:
            logging.error(f"Data visualization error: {str(e)}")
            raise

    def plot_len_dist(self, df, len_col='text_length'):
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=df, x=len_col, bins=50)
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        self._save_plot('text_length_distribution.png')

    def plot_cls_dist(self, df, status_col='status'):
        plt.figure(figsize=self.fig_size)
        sns.countplot(data=df, x=status_col, hue=status_col, palette=self.colors, legend=False)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot('class_distribution.png')

    def plot_len_by_cls(self, df, len_col='text_length', status_col='status'):
        plt.figure(figsize=self.fig_size)
        sns.boxplot(data=df, x=status_col, y=len_col, hue=status_col, palette=self.colors, legend=False)
        plt.title('Text Length by Class')
        plt.xlabel('Class')
        plt.ylabel('Text Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot('length_by_class.png') 