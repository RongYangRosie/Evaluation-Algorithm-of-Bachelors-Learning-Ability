import sklearn
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics


students = load_files("processed_file.csv")

