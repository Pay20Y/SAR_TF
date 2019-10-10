import os
import json

class JSONLoader(object):
    def __init__(self, image_dir, slice_index=-1):
        self.image_dir = image_dir
        self.slice_index = slice_index

    def parse_gt(self, gt_file):
        images_path = []
        transcriptions = []

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_dict = json.load(f)

        for i, k in enumerate(gt_dict.keys()):
            if self.slice_index > 0:
                if i > self.slice_index:
                    break
            item = gt_dict[k]

            images_path.append(os.path.join(self.image_dir, k))
            transcriptions.append(item['transcription'])

        return images_path[:self.slice_index], transcriptions[:self.slice_index]