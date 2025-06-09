import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import numpy as np
from config import FIG_SIZE, WC_SIZE, COLOR_PALETTE, CV_FOLDS
import os
from sklearn.model_selection import learning_curve
import logging

class Visualizer:
    def __init__(self, save_dir='plots'):
        """Initialize visualizer with style settings."""
        plt.style.use('seaborn-v0_8') 
        self.colors = sns.color_palette(COLOR_PALETTE)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def _save_plot(self, filename):
        """Save the current plot to file."""
        if os.path.isabs(filename) or filename.startswith(('plots/', './plots/')):
            save_path = filename
        else:
            save_path = os.path.join(self.save_dir, filename)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_cm(self, y_true, y_pred, labels, title='Confusion Matrix', filename='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=FIG_SIZE)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        self._save_plot(filename)
    
    def plot_metrics(self, metrics_dict, title='Model Performance Comparison', filename='model_performance_comparison.png'):
        try:
            model_names = list(metrics_dict.keys())
            algo_names = list(metrics_dict[model_names[0]].keys())
            metrics_names = ['accuracy', 'precision', 'recall', 'f1']
            
            fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 4*len(metrics_names)))
            if len(metrics_names) == 1:
                axes = [axes]

            for i, metric_name in enumerate(metrics_names):
                ax = axes[i]
                data = []
                labels = []
 
                for model_name in model_names:
                    for algo in algo_names:
                        if metric_name in metrics_dict[model_name][algo]:
                            data.append(metrics_dict[model_name][algo][metric_name])
                            labels.append(f"{model_name}-{algo}")
                
                if data:
                    bars = ax.bar(labels, data)
                    ax.set_title(f'{metric_name.capitalize()} Score')
                    ax.set_ylabel('Score')
                    ax.set_ylim(0, 1.1)

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom')
                    
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.suptitle(title)
            plt.tight_layout()
            self._save_plot(filename)
            
        except Exception as e:
            logging.error(f"Metric plot error: {str(e)}")
            raise
    
    def plot_len_dist(self, df, len_col='len', filename='length_distribution.png'):
        """Plot text length distribution."""
        plt.figure(figsize=FIG_SIZE)
        sns.histplot(data=df, x=len_col, bins=30)
        plt.title('Text Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        self._save_plot(filename)
    
    def plot_cls_dist(self, df, label_col='status', filename='class_distribution.png'):
        """Plot class distribution."""
        plt.figure(figsize=FIG_SIZE)
        sns.countplot(data=df, x=label_col, palette=self.colors)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot(filename)
    
    def plot_len_by_cls(self, df, len_col='len', label_col='status', filename='length_by_class.png'):
        """Plot text length by class."""
        plt.figure(figsize=FIG_SIZE)
        sns.boxplot(data=df, x=label_col, y=len_col, palette=self.colors)
        plt.title('Text Length by Class')
        plt.xlabel('Class')
        plt.ylabel('Length')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_plot(filename)
    
    def plot_wc(self, text, title='Word Cloud', filename='wordcloud.png'):
        """Plot word cloud."""
        wordcloud = WordCloud(width=WC_SIZE[0], height=WC_SIZE[1],
                            background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        self._save_plot(filename)
    
    def plot_learning_curve(self, model, X, y, title='Learning Curve', filename='learning_curve.png'):
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model,
                X,
                y,
                train_sizes=train_sizes,
                cv=CV_FOLDS,
                n_jobs=1,
                scoring='accuracy'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, label='Training score')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.plot(train_sizes_abs, test_mean, label='Cross-validation score')
            plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
            
            plt.title(title)
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            self._save_plot(filename)
            
        except Exception as e:
            logging.error(f"Learning curve plot error: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_names, importance, title='Feature Importance', filename='feature_importance.png'):
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()
        self._save_plot(filename)
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', filename='confusion_matrix.png', labels=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels if labels is not None else 'auto',
                   yticklabels=labels if labels is not None else 'auto')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        self._save_plot(filename) 