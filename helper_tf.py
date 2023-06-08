from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def results(y_true, y_pred):
  """Takes true and predicted values and gives 4 evaluation metrics"""
  
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  accuracy_0 = accuracy_score(y_true, y_pred) * 100
  f1_score_0 = model_f1
  recall_0 = model_recall
  precision_0 = model_precision
  return {"F1_Score" : f1_score_0, "Accuracy" : accuracy_0, "Precision": precision_0,"Recall": recall_0}
