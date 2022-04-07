import os
import numpy as np
import json
from PIL import Image
from matplotlib import pyplot as plt
from utils import *
from multiprocessing import Pool

### Parameters
test_ims = './patches/'
data_path = './RedLights2011_Medium'
preds_path = './predictions' 
os.makedirs(preds_path,exist_ok=True)
threshold = .7
multithreaded = True


def detect_red_light(img, file): 
    kernels = load_kernels(test_ims)
    boxes = []
    for i in range(len(kernels)): 
        kernel = kernels[i]
        cur_boxes = compute_matches(kernel, img, threshold=threshold, similarity=simililarity_mse)
        boxes.extend(cur_boxes)
        print(f'finished kernel: {i}: {file}')
    
    bounding_boxes = refine_boxes(boxes)
 
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes


def prediction(file): 
    I = Image.open(os.path.join(data_path,file))
    I = np.asarray(I)
    pred = detect_red_light(I, file)
    file = file.strip('.jpg')
    with open(os.path.join(preds_path, file + '.json'),'w') as f:
        json.dump(pred,f)
    print(f'done: {file}')

if __name__ == '__main__': 
    file_names = sorted(os.listdir(data_path)) 
    file_names = [f for f in file_names if '.jpg' in f and 'RL-0001.jpg' not in f] 

    if multithreaded:  
        with Pool(32) as p: 
            print(p.map(prediction, file_names))
    else: 
        for file in file_names: 
            prediction(file)
    combine_predictions(preds_path)