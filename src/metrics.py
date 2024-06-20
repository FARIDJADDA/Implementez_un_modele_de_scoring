from sklearn.metrics import make_scorer, confusion_matrix

def business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -(cost_fn * fn + cost_fp * fp)  # Minimisation du coût


def business_score(y_true, y_pred, cost_fn=10, cost_fp=1):
    """
    Calculate the business score for predictions.

    Parameters:
    - y_true: array, true labels.
    - y_pred: array, predicted labels.
    - cost_fn: int, cost associated with a false negative.
    - cost_fp: int, cost associated with a false, positive.

    Returns:
    - float, normalized business score between 0 and 1.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = cost_fn * fn + cost_fp * fp
    max_cost = cost_fn * (fn + tp) + cost_fp * (fp + tn)  # worst case scenario: all predictions are wrong
    
    # Normalize and subtract from 1 to flip the scale: higher is better
    return 1 - (total_cost / max_cost)