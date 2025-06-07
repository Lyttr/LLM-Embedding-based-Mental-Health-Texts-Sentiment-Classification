import logging
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from config import FIG_SIZE, WC_SIZE, PLOT_STYLE, COLOR_PALETTE

class DataVisualizer:

    def __init__(self):

        plt.style.use(PLOT_STYLE)
        self.fig_size = FIG_SIZE
        self.color_palette = COLOR_PALETTE

    def visualize_data(self, df, output_dir='plots/data_analysis'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            df['text_length'] = df['statement'].str.len()
            logging.info("Generating data visualizations...")
            self.plot_len_dist(df, len_col='text_length', 
                             filename=f'{output_dir}/text_length_distribution.png')
            logging.info("Saved text length distribution plot")
            self.plot_cls_dist(df, status_col='status',
                             filename=f'{output_dir}/class_distribution.png')
            logging.info("Saved class distribution plot")
            self.plot_len_by_cls(df, len_col='text_length', status_col='status',
                               filename=f'{output_dir}/length_by_class.png')
            logging.info("Saved length by class plot")
            
        except Exception as e:
            logging.error(f"Data visualization error: {str(e)}")
            raise

    def plot_len_dist(self, df, len_col='text_length', filename='text_length_distribution.png'):
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=df, x=len_col, bins=50)
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        plt.savefig(filename)
        plt.close()

    def plot_cls_dist(self, df, status_col='status', filename='class_distribution.png'):
        plt.figure(figsize=self.fig_size)
        sns.countplot(data=df, x=status_col)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_len_by_cls(self, df, len_col='text_length', status_col='status', 
                       filename='length_by_class.png'):
        plt.figure(figsize=self.fig_size)
        sns.boxplot(data=df, x=status_col, y=len_col)
        plt.title('Text Length by Class')
        plt.xlabel('Class')
        plt.ylabel('Text Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() 