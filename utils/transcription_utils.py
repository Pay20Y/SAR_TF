import string
import editdistance
import numpy as np
from data_provider.data_utils import get_vocabulary

def idx2label(inputs, id2char=None, char2id=None):

    if id2char is None:
        voc, char2id, id2char = get_vocabulary(voc_type="ALLCASES_SYMBOLS")

    def end_cut(ins):
        cut_ins = []
        for id in ins:
            if id != char2id['EOS']:
                if id != char2id['UNK']:
                    cut_ins.append(id2char[id])
            else:
                break
        return cut_ins

    if isinstance(inputs, np.ndarray):
        assert len(inputs.shape) == 2, "input's rank should be 2"
        results = [''.join([ch for ch in end_cut(ins)]) for ins in inputs]
        return results
    else:
        print("input to idx2label should be numpy array")
        return inputs

def calc_metrics(predicts, labels, metrics_type='accuracy'):
    assert metrics_type in ['accuracy', 'editdistance'], "Unsupported metrics type {}".format(metrics_type)

    if metrics_type == 'accuracy':
        acc_list = [(pred == tar) for pred, tar in zip(predicts, labels)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        return accuracy
    elif metrics_type == 'editdistance':
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(predicts, labels)]
        eds = sum(ed_list)
        return eds

    return -1
