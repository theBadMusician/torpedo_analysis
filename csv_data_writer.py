#! /usr/bin/env python3

import glob
import os
import csv

txt_names = glob.glob(os.getcwd() + "/videos/*.txt")

point_dict = {}
for txt_fn in txt_names:
    slash_idx = txt_fn.rfind('/')
    b_slash_idx = txt_fn.rfind('\\')
    name_start_idx = max(slash_idx, b_slash_idx)

    run_name = txt_fn[name_start_idx + 1:-4]
    with open(txt_fn, 'r') as file:
        txt = file.readlines()
        last_line = txt[-1:]
        point_dict[run_name] = last_line

with open(os.getcwd() + "/video_detections.csv", 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    header = []
    vals = []
    for fn, point_val in point_dict.items():
        header.append(fn)
        vals.append(float(point_val[0][:-2]))

    writer.writerows((header, vals))
    
        
