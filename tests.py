from heapq import merge
import numpy as np
from utils import *
from PIL import Image

data_path = '../RedLights2011_Medium'
test_path = '../tests'

def convolution_tests(): 
    kernel = np.ones((1,1))
    mat = np.ones((5,5))
    out = convolve(kernel, mat)
    if (out == np.ones((5,5))).all(): 
        print("passed convolution test 1")
    out = convolve(kernel, mat, padding=1, fill=1)
    if (out == np.ones((7,7))).all(): 
        print("passed convolution test 2")
    out = convolve(kernel, mat, padding=1, fill=0)
    test = np.pad(out, 0, constant_values=0)
    if (out == test).all(): 
        print("passed convolution test 3")
    kernel = np.ones((2,2))
    out = convolve(kernel, mat)
    if (out == np.ones((4,4))*4).all(): 
        print("passed convolution test 4")
    kernel = np.ones((3,3))
    out = convolve(kernel, mat)
    if (out == np.ones((3,3))*9).all(): 
        print("passed convolution test 5")
    mat = np.ones((5,10))
    kernel = np.ones((2,2))
    out = convolve(kernel, mat)
    if (out == np.ones((4,9))*4).all(): 
        print("passed convolution test 6")
    kernel = np.ones((3,3))
    out = convolve(kernel, mat)
    if (out == np.ones((3,8))*9).all(): 
        print("passed convolution test 7")

def identity_conv(mat, save_dir='../tests/'): 
    kernel = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])
    mat[:,:,0] = convolve(kernel, mat[:,:,0], padding=1)
    mat[:,:,1] = convolve(kernel, mat[:,:,1], padding=1)
    mat[:,:,2] = convolve(kernel, mat[:,:,2], padding=1)
    img = Image.fromarray(np.uint8(mat)).convert('RGB')
    img.save(save_dir + 'identity.jpg')

def blur_conv(mat, save_dir='../tests/'): 
    kernel = np.array([[.0626, .125, .0625],[.125, .25, .125],[.0626, .125, .0625]])
    mat[:,:,0] = convolve(kernel, mat[:,:,0], padding=1)
    mat[:,:,1] = convolve(kernel, mat[:,:,1], padding=1)
    mat[:,:,2] = convolve(kernel, mat[:,:,2], padding=1)
    img = Image.fromarray(np.uint8(mat)).convert('RGB')
    img.save(save_dir + 'blur.jpg')

def sharpen_conv(mat, save_dir='../tests/'): 
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    mat[:,:,0] = convolve(kernel, mat[:,:,0], padding=1)
    mat[:,:,1] = convolve(kernel, mat[:,:,1], padding=1)
    mat[:,:,2] = convolve(kernel, mat[:,:,2], padding=1)
    img = Image.fromarray(np.uint8(mat)).convert('RGB')
    img.save(save_dir + 'sharpen.jpg')
    
def conv_visual_tests(): 
    file = '../RedLights2011_Medium/RL-001.jpg'
    
    mat = np.asarray(img)
    identity_conv(mat)
    blur_conv(mat)
    sharpen_conv(mat)
def merge_box_tests(): 
    # test = [[100, 200, 150, 250], [100, 200, 150, 250]]
    # test_box(test)
    # test = [[150, 159, 141, 150], [147, 150, 134, 130]]
    # test_box(test)
    test = [[150, 159, 141, 150], [323, 162, 314, 153], [115, 175, 106, 166], [132, 189, 123, 180], [76, 190, 67, 181], [77, 190, 68, 181], [78, 190, 69, 181], [142, 190, 133, 181], [77, 191, 68, 182], [78, 191, 69, 182], [145, 196, 136, 187], [145, 197, 136, 188], [427, 199, 418, 190], [427, 200, 418, 191], [415, 228, 406, 219], [415, 229, 406, 220], [217, 244, 208, 235], [79, 191, 66, 178], [78, 192, 65, 179], [79, 192, 66, 179], [80, 192, 67, 179], [78, 193, 65, 180], [79, 193, 66, 180], [80, 193, 67, 180], [79, 194, 66, 181], [80, 194, 67, 181], [147, 199, 134, 186], [429, 201, 416, 188], [429, 202, 416, 189], [323, 161, 314, 152], [323, 162, 314, 153], [323, 163, 314, 154] ] 
    test_box(test)

def test_box(boxes): 
    file = '../RedLights2011_Medium/RL-001.jpg'
    img = Image.open(file)
    print(len(boxes))
    draw_boxes(img, boxes)
    img.show()
    img = Image.open(file)
    boxes = refine_boxes(boxes)
    print(len(boxes))
    draw_boxes(img, boxes)
    img.show()
if __name__ == "__main__": 
    # convolution_tests()
    # conv_visual_tests()
    merge_box_tests()