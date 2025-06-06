import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from config import FIGURE_SIZE, WORDCLOUD_SIZE

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_cm(self, cm, labels, title='Confusion Matrix'):
        """Plot confusion matrix with labels."""
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels.keys(), yticklabels=labels.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_len_dist(self, df):
        """Plot distribution of text lengths."""
        plt.figure(figsize=(8, 5))
        sns.histplot(df['len'], bins=40, kde=True)
        plt.title('Text Length Distribution')
        plt.xlabel('Words')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def plot_cls_dist(self, df):
        """Plot distribution of sentiment classes."""
        plt.figure(figsize=FIGURE_SIZE)
        sns.countplot(data=df, x='status', order=df['status'].value_counts().index)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_len_by_cls(self, df):
        """Plot text length distribution by sentiment class."""
        plt.figure(figsize=FIGURE_SIZE)
        sns.boxplot(data=df, x='status', y='len')
        plt.title('Length by Class')
        plt.xlabel('Class')
        plt.ylabel('Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_wc(self, texts):
        """Generate and display word cloud from texts."""
        text = " ".join(texts)
        
        wc = WordCloud(
            width=WORDCLOUD_SIZE[0],
            height=WORDCLOUD_SIZE[1],
            background_color='white',
            max_words=200,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud", fontsize=16)
        plt.tight_layout()
        plt.show() 