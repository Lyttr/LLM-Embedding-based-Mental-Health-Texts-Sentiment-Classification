import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import numpy as np
from config import FIG_SIZE, WC_SIZE, PLOT_STYLE, COLOR_PALETTE

class Visualizer:
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.colors = sns.color_palette(COLOR_PALETTE)
    
    def plot_cm(self, y_true, y_pred, labels, title='Confusion Matrix'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=FIG_SIZE)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_metrics(self, metrics_dict, title='Model Performance Comparison'):
        """Plot performance metrics comparison."""
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        x = np.arange(len(models))
        width = 0.8 / len(metrics)
        
        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(x + width*len(metrics)/2, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_len_dist(self, df, len_col='len'):
        """Plot text length distribution."""
        plt.figure(figsize=FIG_SIZE)
        sns.histplot(data=df, x=len_col, bins=30)
        plt.title('Text Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.show()
    
    def plot_cls_dist(self, df, label_col='label'):
        """Plot class distribution."""
        plt.figure(figsize=FIG_SIZE)
        sns.countplot(data=df, x=label_col, palette=self.colors)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_len_by_cls(self, df, len_col='len', label_col='label'):
        """Plot text length by class."""
        plt.figure(figsize=FIG_SIZE)
        sns.boxplot(data=df, x=label_col, y=len_col, palette=self.colors)
        plt.title('Text Length by Class')
        plt.xlabel('Class')
        plt.ylabel('Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_wc(self, text, title='Word Cloud'):
        """Plot word cloud."""
        wordcloud = WordCloud(width=WC_SIZE[0], height=WC_SIZE[1],
                            background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    def plot_learning_curve(self, train_sizes, train_scores, test_scores, title='Learning Curve'):
        """Plot learning curve."""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=FIG_SIZE)
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.title(title)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance, title='Feature Importance'):
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, title='ROC Curve'):
        """Plot ROC curve."""
        plt.figure(figsize=FIG_SIZE)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show() 