import numpy as np
from typing import List


def get_change_idx(labels: np.ndarray):
    return np.where(labels == 1)[0]


def binary_evaluation(gt: np.ndarray, pd: np.ndarray, tolerance_interval: int):
    """
    Parameters
    ----------
    gt: ground truth labels of 0,1
    pd: predicted labels of 0,1,
    tolerance_interval: half length of tolerance window, 
        tolerance interval is set to 3,
        and a change gt is at time 6,
        then a change pd at 3-9 is considered as valid detection
    """
    gt_idx = get_change_idx(gt)
    pd_idx = get_change_idx(pd)
    gt_start = np.maximum(0, gt_idx-tolerance_interval).reshape((-1, 1))
    gt_end = np.minimum(gt_idx+tolerance_interval+1, len(gt)).reshape((-1, 1))
    gt_boundaries = np.hstack([gt_start, gt_end])
    pd_start = np.maximum(0, pd_idx-tolerance_interval).reshape((-1, 1))
    pd_end = np.minimum(pd_idx+tolerance_interval+1, len(pd)).reshape((-1, 1))
    pd_boundaries = np.hstack([pd_start, pd_end])

    tp = 0
    for boundary in gt_boundaries:
        num_detected = np.sum(pd[boundary[0]:boundary[1]])
        if num_detected > 1:
            num_detected = 1
        tp += num_detected

    # fn can be calculated as actual change points not in predicted segments
    fn = 0
    start = 0
    for idx in range(len(pd_boundaries)):
        end = pd_boundaries[idx][0]
        num_omitted = np.sum(gt[start:end])
        fn += num_omitted
        start = pd_boundaries[idx][1]
    if start != len(pd):
        end = len(pd)
        num_omitted = np.sum(gt[start:end])
        fn += num_omitted

    # fn can be calculated as change points omitted in ground-truth segments
    # fn_val = 0
    # for boundary in gt_boundaries:
    #     num_detected = np.sum(pd[boundary[0]:boundary[1]])
    #     if num_detected == 0:
    #         fn_val += 1

    fp = 0
    start = 0
    for idx in range(len(gt_boundaries)):
        end = gt_boundaries[idx][0]
        num_false_alarm = np.sum(pd[start:end])
        fp += num_false_alarm
        start = gt_boundaries[idx][1]
    if start != len(gt):
        end = len(gt)
        num_false_alarm = np.sum(pd[start:end])
        fp += num_false_alarm

    precision = 1.0*tp/np.sum(pd)
    recall = 1.0*tp/np.sum(gt)
    f1 = (2*precision*recall)/(precision+recall)
    missing_rate = 1.0*fn/np.sum(gt)

    return precision, recall, f1, missing_rate


def add_evaluation(gt: np.ndarray, pd: np.ndarray, tolerance_interval: int):
    """
    Parameters
    ----------
    gt: ground truth labels of 0,1
    pd: predicted labels of 0,1,
    tolerance_interval: half length of tolerance window, 
        tolerance interval is set to 3,
        and a change gt is at time 6,
        then a change pd at 3-9 is considered as valid detection
    """
    gt_idx = get_change_idx(gt)
    pd_idx = get_change_idx(pd)
    gt_start = np.maximum(0, gt_idx-tolerance_interval).reshape((-1, 1))
    gt_end = np.minimum(gt_idx+tolerance_interval+1, len(gt)).reshape((-1, 1))
    gt_boundaries = np.hstack([gt_start, gt_end])
    pd_start = np.maximum(0, pd_idx-tolerance_interval).reshape((-1, 1))
    pd_end = np.minimum(pd_idx+tolerance_interval+1, len(pd)).reshape((-1, 1))
    pd_boundaries = np.hstack([pd_start, pd_end])

    add = 0
    total_detected = 0
    for boundary in gt_boundaries:
        pd_subseq = pd[boundary[0]:boundary[1]]
        num_detected = np.sum(pd_subseq)
        if num_detected >= 1:
            num_detected = 1
            total_detected += num_detected
            cp_idx = np.where(pd_subseq == 1)[0][0]
            add += cp_idx-tolerance_interval
    add = 1.0*add/total_detected

    fn = 0
    start = 0
    for idx in range(len(pd_boundaries)):
        end = pd_boundaries[idx][0]
        num_omitted = np.sum(gt[start:end])
        fn += num_omitted
        start = pd_boundaries[idx][1]
    if start != len(pd):
        end = len(pd)
        num_omitted = np.sum(gt[start:end])
        fn += num_omitted
    missing_rate = 1.0*fn/np.sum(gt)

    fp = 0
    start = 0
    for idx in range(len(gt_boundaries)):
        end = gt_boundaries[idx][0]
        num_false_alarm = np.sum(pd[start:end])
        fp += num_false_alarm
        start = gt_boundaries[idx][1]
    if start != len(gt):
        end = len(gt)
        num_false_alarm = np.sum(pd[start:end])
        fp += num_false_alarm

    return add, missing_rate


def nab_evaluation(gt: np.ndarray, pd: np.ndarray, tolerance_interval: int):
    """
    Parameters
    ----------
    gt: ground truth labels of 0,1
    pd: predicted labels of 0,1,
    tolerance_interval: half length of tolerance window, 
        tolerance interval is set to 3,
        and a change gt is at time 6,
        then a change pd at 3-9 is considered as valid detection
    """

    A_TP = 1.5
    A_FP = -0.5
    A_TN = 1.5
    A_FN = -1.5

    def score_function(relative_position):
        """
        score function is shaped like 
            (A_TP-A_FN)*(1+e^(\beta*relative_position))-1
        ensure A_TP-A_FN==2 and 
            \beta is a hyper-parameter adjusted with toleance_interval
        """
        return (A_TP-A_FN)*(1.0/(1+np.exp(0.7*relative_position)))-1

    gt_idx = get_change_idx(gt)
    pd_idx = get_change_idx(pd)
    gt_start = np.maximum(0, gt_idx-tolerance_interval).reshape((-1, 1))
    gt_end = np.minimum(gt_idx+tolerance_interval+1, len(gt)).reshape((-1, 1))
    gt_boundaries = np.hstack([gt_start, gt_end])
    pd_start = np.maximum(0, pd_idx-tolerance_interval).reshape((-1, 1))
    pd_end = np.minimum(pd_idx+tolerance_interval+1, len(pd)).reshape((-1, 1))
    pd_boundaries = np.hstack([pd_start, pd_end])

    score = 0.0
    if len(gt_idx) == 0:  # if no change in signal, then every predicted changes are false positive
        score = np.sum(pd)*A_FP
    else:
        if len(pd_idx) == 0:  # if prediction missed all change, then every actual changes are false negative
            score = np.sum(gt)*A_FN
        else:
            for boundary in gt_boundaries:
                pd_subseq = pd[boundary[0]:boundary[1]]
                if np.sum(pd_subseq) == 0:  # if missed current change, report false negative
                    score += A_FN
                else:
                    cp_idx = np.where(pd_subseq == 1)[0][0]
                    relative_position = cp_idx-tolerance_interval
                    score += score_function(relative_position)

            # every predicted change not in gt bounaries are false alarm
            num_fp = 0
            start = 0
            for idx in range(len(gt_boundaries)):
                end = gt_boundaries[idx][0]
                num_false_alarm = np.sum(pd[start:end])
                num_fp += num_false_alarm
                start = gt_boundaries[idx][1]
            if start != len(gt):
                end = len(gt)
                num_false_alarm = np.sum(pd[start:end])
                num_fp += num_false_alarm

            score += num_fp*A_FP

    perfect_score = A_TP*np.sum(gt)
    worst_score = A_FN*np.sum(gt)

    scaled_score = (score-worst_score)/(perfect_score-worst_score)

    return scaled_score.item()
