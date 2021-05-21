import os
from os.path import join
from matplotlib.cm import get_cmap
from numpy.core.numeric import outer
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from joblib import load, dump

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class RunIrisClassification():
    """Class to generate data and test classifiers using a train-test split
    """
    def __init__(self):
        self.output_dir = join('..', 'clf_results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.clf_dict = {
            'DecisionTree': DecisionTreeClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier()
        }
        self.results = pd.DataFrame()

    def load_data(self):
        """Loads iris data from sklearn
        """
        self.data = load_iris()
        self.df = pd.DataFrame(data=self.data.data, columns=self.data.feature_names)
        self.df["target"] = self.data.target

    def split_data(self):
        """Populates X and y with train and test data
        """
        df_to_split = self.df.copy(deep=True)
        self.y = df_to_split['target']
        self.X = df_to_split.drop(columns=['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.33, 
            random_state=42, 
            stratify=self.y
        )

    def train_models(self):
        """Trains classifiers in clf dict
        """
        for label, clf in self.clf_dict.items():
            clf_output_dir = join(self.output_dir, label)
            os.makedirs(clf_output_dir, exist_ok=True)
            clf.fit(self.X_train, self.y_train)
            dump(clf, join(clf_output_dir, 'model.pkl'))
            self.get_metrics(clf, clf_output_dir, train_or_test='Train')
            self.get_metrics(clf, clf_output_dir, train_or_test='Test')

    def get_metrics(self, clf, clf_output_dir, train_or_test='Train'):
        if train_or_test.lower() == 'train':
            X = self.X_train
            y = self.y_train
        else:
            X = self.X_test
            y = self.y_test
        preds = clf.predict(X)        
        cm_plt = plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues)
        cm_plt.figure_.savefig(join(clf_output_dir, f'{train_or_test}_confusion_matrix.png'))
        report = classification_report(y, preds, output_dict=True)
        df_report = pd.DataFrame(report)
        df_report.to_csv(join(clf_output_dir, f'{train_or_test}_metrics.csv'))


if __name__=='__main__':
    ric = RunIrisClassification()
    ric.load_data()
    ric.split_data()
    ric.train_models()