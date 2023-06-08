from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def results(y_true, y_pred):
  """Takes true and predicted values and gives 4 evaluation metrics"""
  
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  accuracy_0 = accuracy_score(y_true, y_pred)
  f1_score_0 = f1_score(y_true, y_pred)
  recall_0 = recall_score(y_true, y_pred)
  precision_0 = precision_score(y_true, y_pred)
  return {"F1_Score" : f1_score_0, "Accuracy" : accuracy_0, "Precision": precision_0,"Recall": recall_0}
