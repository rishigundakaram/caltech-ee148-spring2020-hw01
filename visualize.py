import json
from PIL import Image, ImageDraw
from utils import draw_boxes

prediction_path = "../predictions/pred.json"
out_path = "../out/"
data_path = "../RedLights2011_Medium/"
f = open(prediction_path)
boxes = json.load(f)

for file in boxes.keys(): 
    jpg = Image.open(data_path + file)
    draw_boxes(jpg, boxes[file])
    jpg.save(out_path + file)    



