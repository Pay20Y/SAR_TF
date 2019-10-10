import numpy as np

def accuracy(predicts, targets, case_sens=False):
    if not case_sens:
        predicts = [pred.lower() for pred in predicts]
        targets = [targ.lower() for targ in targets]

    accuracy_list = [pred == targ for pred, targ in zip(predicts, targets)]
    acc_rate = 1.0 * sum(accuracy_list) / len(accuracy_list)

    return acc_rate
