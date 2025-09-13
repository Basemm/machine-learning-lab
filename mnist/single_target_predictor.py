import numpy as np
from common import get_dataset
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt


class SingleMNISTTargetPredictor:
  def __init__(self, target):
    self.model = SGDClassifier(random_state=42)
    self.target = str(target)
    self.is_fitted = False

    X_train, y_train, X_test, y_test = get_dataset()
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

    self.y_train_target = self.y_train == self.target
    self.y_train_target_scores = np.array([])


  def fit(self):
    self.model.fit(self.X_train, self.y_train_target)
    self.is_fitted = True

    return self

  def predict(self, X):
    if not self.is_fitted:
      self.fit()

    return self.model.predict(X)

  def predict_at_precision(self, X, precision):
    self.cross_val_predict()

    y_scores = self.model.decision_function(X)
    precisions, _, thresholds = precision_recall_curve(self.y_train_target, self.y_train_target_scores)

    threshold = thresholds[(precisions >= precision).argmax()]

    print(f"Threshold at {precision} precision: {threshold}")

    return y_scores >= threshold

  def predict_at_recall(self, X, recall):
    self.cross_val_predict()

    y_scores = self.model.decision_function(X)
    _, recalls, thresholds = precision_recall_curve(self.y_train_target, self.y_train_target_scores)

    threshold = thresholds[(recalls >= recall).argmax()]

    print(f"Threshold at {recall} recall: {threshold}")

    return y_scores >= threshold

  def cross_val_predict(self):
    if not self.is_fitted:
      self.fit()

    if self.y_train_target_scores.size == 0:
      self.y_train_target_scores = cross_val_predict(self.model, self.X_train, self.y_train_target, cv=3,
                                  method="decision_function")

    return self.y_train_target_scores


  def precision_score(self):
    if not self.is_fitted:
      self.fit()

    if self.y_train_target_scores.size == 0:
      self.y_train_target_scores = cross_val_predict(self.model, self.X_train, self.y_train_target, cv=3,
                                 method="decision_function")

    precision_score = precision_score(self.y_train_target, self.y_train_target_scores)

    return precision_score

  def recall_score(self):
    self.cross_val_predict()

    recall_score = recall_score(self.y_train_target, self.y_train_target_scores)

    return recall_score

  def confusion_matrix(self):
    self.cross_val_predict()


    y_train_pred = self.y_train_target_scores >= 0

    return confusion_matrix(self.y_train_target, y_train_pred)

  def plot_precision_recall_curve(self):
    self.cross_val_predict()

    precisions, recalls, _ = precision_recall_curve(self.y_train_target, self.y_train_target_scores)

    disp = PrecisionRecallDisplay(precision=precisions, recall=recalls)
    disp.plot()

    return disp

  def plot_roc_curve(self):
    self.cross_val_predict()

    fpr, tpr, _ = roc_curve(self.y_train_target, self.y_train_target_scores)

    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name="SGDClassifier")
    disp.plot()

    return disp

  def plot_confusion_matrix(self):
    self.cross_val_predict()

    y_train_pred = self.y_train_target_scores >= 0

    cm = confusion_matrix(self.y_train_target, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    return disp

  def plot_thresholds_precision_recall(self):
    self.cross_val_predict()

    precisions, recalls, thresholds = precision_recall_curve(self.y_train_target, self.y_train_target_scores)


    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

    plt.show()
