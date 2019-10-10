import os

class TextLoader(object):
    def __init__(self, image_dir, slice_index=-1):
        self.image_dir = image_dir
        self.slice_index = slice_index
    def parse_gt(self, gt_file):
        images_path = []
        transcriptions = []

        with open(gt_file, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                if self.slice_index > 0:
                    if i > self.slice_index:
                        break
                line = line.strip().split()
                if len(line) != 2:
                    continue
                images_path.append(os.path.join(self.image_dir, line[0]))
                transcriptions.append(line[1])

        return images_path[:self.slice_index], transcriptions[:self.slice_index]