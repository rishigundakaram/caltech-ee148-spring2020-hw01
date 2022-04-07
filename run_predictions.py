import os
import numpy as np
import json
from PIL import Image
from matplotlib import pyplot as plt
from utils import *
from multiprocessing import Pool

test_ims = '../patches/'
threshold = .7
def alg_1(img): 
    bounding_boxes = []
    red = img[:, :, 0]
    diameters = [i for i in range(6,8)]
    for i in diameters:
        kernel = circle_kernel(i)
        out = convolve(kernel, red)
        img = Image.fromarray(out)
        img.show()
    return bounding_boxes

def detect_red_light(img, file): 
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
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

    
# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../predictions' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f and 'RL-0001.jpg' not in f] 

def prediction(file): 
    I = Image.open(os.path.join(data_path,file))
    I = np.asarray(I)
    pred = detect_red_light(I, file)
    file = file.strip('.jpg')
    with open(os.path.join(preds_path, file + '.json'),'w') as f:
        json.dump(pred,f)
    print(f'done: {file}')

if __name__ == '__main__': 
    with Pool(8) as p: 
        print(p.map(prediction, file_names))