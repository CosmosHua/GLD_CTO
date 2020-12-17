#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, json
import os.path as osp
import argparse, uuid
from PIL import Image
from glob import glob
from datetime import datetime
from shutil import copy, copyfile
from collections import defaultdict
from tqdm import tqdm

import labelme
try: import pycocotools.mask
except ImportError:
    print('Please: pip install pycocotools\n'); sys.exit(1)


#######################################################
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args(); output_dir = args.output_dir

    print('Creating dataset:', output_dir)
    assert not osp.exists(output_dir); os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, 'Images'))

    now = datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f') ),
        licenses=[dict(url=None, id=0, name=None)],
        images=[], # license, url, file_name, height, width, date_captured, id
        type='instances',
        annotations=[], # segmentation, area, iscrowd, image_id, bbox, category_id, id
        categories=[], # supercategory, id, name
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i-1 # start with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name=='_background_'; continue
        class_name_to_id[class_name] = class_id
        d = class_name.find("_") # rfind("_")
        supcat = class_name[:d] if d>0 else class_name
        data['categories'].append(
            dict(supercategory=supcat, id=class_id, name=class_name) )

    ann_file = osp.join(output_dir, 'annotations.json')
    label_files = glob(osp.join(args.input_dir, '*.json'))
    for image_id, jsonfile in enumerate(tqdm(label_files)):
        #print('Generating dataset from:', jsonfile)
        label_file = labelme.LabelFile(filename=jsonfile)
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        
        dst_img = osp.splitext(osp.basename(jsonfile))[0]+".jpg"
        dst_img = osp.join(output_dir, 'Images', dst_img)
        if label_file.imagePath.endswith(".jpg"):
            #copy(osp.join(args.input_dir, label_file.imagePath), dst_img)
            copyfile(osp.join(args.input_dir, label_file.imagePath), dst_img)
        else: Image.fromarray(img).save(dst_img)
        #dst_img = osp.relpath(dst_img, osp.dirname(ann_file)).replace("\\","/")
        #dst_img = osp.relpath(dst_img, output_dir).replace("\\","/")
        dst_img = osp.basename(dst_img)
        
        data['images'].append(
            dict(license=0,
                url=None,
                file_name= dst_img,
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id) )
    
        masks = {}                         # for area
        segmentations = defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_type')
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type )

            if group_id is None: group_id = uuid.uuid1()
            instance = (label, group_id)
            if instance in masks:
                masks[instance] = masks[instance] | mask
            else: masks[instance] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id: continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(
                dict(id=len(data['annotations']),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0) )
    
    with open(ann_file,'w+') as ff:
        json.dump(data, ff, indent=4)


#######################################################
if __name__ == '__main__':
    main()
