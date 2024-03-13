import glob
import os
from pathlib import Path

from reportlab.lib import yaml
from ultralytics.utils.downloads import download


def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


    for fname in glob.glob('/home/nurullah/Masaüstü/copy/copy_labels/*.txt'):
        input_file_path = fname
        output_path =os.path.basename(input_file_path).replace('.txt','')
        (dir / 'labels'/output_path).mkdir(parents=True, exist_ok=True)  # make labels directory
        pbar = tqdm((dir / 'annotations'/output_path).glob('*.txt'), desc=f'Converting {dir}')
        for f in pbar:
            print(f)
            img_size = Image.open((dir / 'sequences' /output_path/ f.name).with_suffix('.jpg')).size
            print(img_size)
            lines = []
            with open(f, 'r') as file:  # read annotation.txt
                for row in [x.split(',') for x in file.read().strip().splitlines()]:
                    if row[4] == '0':  # VisDrone 'ignored regions' class 0
                        continue
                    cls = int(row[5]) - 1
                    box = convert_box(img_size, tuple(map(int, row[:4])))
                    lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                    with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                        fl.writelines(lines)  # write label.txt




yaml_file_path = '/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone.yaml'
yaml_dir_path = Path(yaml_file_path).parent

for d in ['VisDrone2019-VID-val']:#, 'VisDrone2019-VID-val', 'VisDrone2019-VID-test-dev']
    visdrone2yolo(yaml_dir_path / d)

