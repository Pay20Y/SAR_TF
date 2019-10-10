import string
import numpy as np
import cv2
import math

def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):

    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def rotate_img(img, angle, scale=1):
    H, W, _ = img.shape
    rangle = np.deg2rad(angle)  # angle in radians
    new_width = (abs(np.sin(rangle) * H) + abs(np.cos(rangle) * W)) * scale
    new_height = (abs(np.cos(rangle) * H) + abs(np.sin(rangle) * W)) * scale

    rot_mat = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_width - W) * 0.5, (new_height - H) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))),
                             flags=cv2.INTER_LANCZOS4)

    return rot_img