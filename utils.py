import numpy as np
import os
from PIL import Image, ImageDraw
import json
color_threshold = 1
circle_threshold = .5
intensity_threshold = 1


def convolve(kernel, arr, padding=0, fill=0, similarity=None): 
    k_m, k_n = np.shape(kernel)
    arr_m, arr_n = np.shape(arr)
    if padding != 0: 
        arr = np.pad(arr, padding, constant_values=fill)
    out = []
    start_row = k_m//2
    end_row = arr_m + 2*padding - start_row
    if k_m % 2 == 0: end_row += 1 
    start_col = k_n//2
    end_col = arr_n + 2*padding - start_col
    if k_m % 2 == 0: end_col += 1
    for i in range(start_row, end_row): 
        for j in range(start_col, end_col): 
            if k_m % 2 == 1: 
                sub_arr = arr[i-start_row:i+start_row+1,j-start_col:j+start_col+1]
            else: 
                sub_arr = arr[i-start_row:i+start_row,j-start_col:j+start_col]
            if not similarity: 
                weight = (sub_arr*kernel).sum()
            else: 
                weight = similarity(sub_arr, kernel)
            out.append(weight)
    out = np.reshape(out, (arr_m-k_m+1+2*padding, arr_n-k_n+1+2*padding))
    return out

def l2_norm(x,y): 
    return np.square(x, y).sum()

def ssim(x, y, printer=False): 
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2 
    cov = np.cov(x.flatten(), y.flatten())
    num = (2 * np.mean(x)*np.mean(y) + c1)*(2 * cov[0,1] + c2)
    denom = (np.mean(x)**2 + np.mean(y)**2 + c1)*(cov[0,0] + cov[1,1] + c2)
    return num/denom

def l1_norm(x,y): 
    return np.abs(x-y).sum()

def convolve_3d(kernel, arr, padding=0, fill=0): 
    final = np.zeros(np.shape(arr))
    for i in range(3): 
        final[:, :, i] = convolve(kernel[:, :, i], arr[:, :, i], padding=padding, 
                                  similarity=ssim)
    # print(final[:,:,0])
    # np.savetxt('final.csv', np.round(final[:,:,0], 2), delimiter=',')
    # num_elems = np.prod(np.shape(kernel))
    # return np.sum(final, axis=2) / num_elems
    return np.mean(final, axis=2)

def compute_matches(kernel, image, threshold, similarity): 
    padding = np.shape(kernel[:, :, 0])[0] // 2
    convolved = convolve_3d(kernel, image, padding)
    idxs = np.argwhere(convolved > threshold)
    convolved
    boxes = []
    for idx in idxs:
        radius = int(np.shape(kernel[:, :, 0])[0] // 2)
        cur_box = [int(idx[1]) + radius, int(idx[0]) + radius, int(idx[1]) - radius - 1, int(idx[0]) - radius - 1]
        boxes.append(cur_box)
    return boxes

def draw_boxes(image, bounds): 
    draw = ImageDraw.Draw(image)
    for bounding_box in bounds: 
        draw.rectangle(bounding_box)


def circle_kernel(diameter=2): 
    x = [2*i/(diameter) -1 + 1/(diameter) for i in range(diameter)]
    y = [2*i/(diameter) -1 + 1/(diameter) for i in range(diameter)]
    kernel = []
    for i in x: 
        for j in y: 
            if i**2 + j**2 <= circle_threshold: 
                kernel.append(1)
            else: 
                kernel.append(0)
    kernel = np.reshape(kernel, (diameter,diameter))
    kernel = kernel/kernel.sum()
    kernel = np.round(kernel, 3)
    return kernel

def preprocess(img): 
    c1 = img[:, :, 0] > color_threshold*img[:, :, 1]
    c2 = img[:, :, 0] > color_threshold*img[:, :, 2]
    c3 = img[:,:,0] > 100
    mask = c1 & c2 & c3
    img[:,:,0] *= mask 
    return img

def simililarity_mse(mat, kernel):
    return np.square(mat - kernel).mean()

def load_kernels(path): 
    files = os.listdir(path)
    kernels = []
    for file in files: 
        if file == '.DS_Store': 
            continue
        I = Image.open(path + file)
        cur = np.asarray(I)
        kernels.append(cur)
    return kernels

def overlap(box1, box2): 
    return not (box1[2] >= box2[0] or  
                    box1[3] >= box2[1] or  
                    box1[0] <= box2[2] or  
                    box1[1] <= box2[3]) 

def merge_boxes(box1, box2): 
    return [min(box1[0], box2[0]), 
         min(box1[1], box2[1]), 
         max(box1[2], box2[2]),
         max(box1[3], box2[3])]

def refine_boxes(boxes): 
    if not len(boxes): 
        return []
    final_boxes = []
    start = 0
    next = 0
    while start < len(boxes): 
        cur_box = boxes[start]
        next = start + 1
        while next < len(boxes): 
            if overlap(cur_box, boxes[next]):
                cur_box = merge_boxes(cur_box, boxes[next])
                del boxes[next]
                continue
            next += 1
        start += 1
        final_boxes.append(cur_box)
    return final_boxes

def combine_predictions(path): 
    predictions = os.listdir(path)
    predictions = [i for i in predictions if '.json' in i]
    preds = {}
    for cur in predictions: 
        print(path + '/' + cur)
        with 
        cur_pred = json.load(path + '/' + cur)
        print(cur_pred)
        for key in cur_pred.keys(): 
            preds[key] = cur_pred[key]
    with open(path + '/pred.json', 'w') as f: 
        json.dump(preds, f)

if __name__ == '__main__': 
    path = '../predictions'
    combine_predictions(path)