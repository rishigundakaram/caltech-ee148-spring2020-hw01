import json
from PIL import Image, ImageDraw
from utils import draw_boxes

prediction_path = "../predictions/pred.json"
save_path = "../report/"
data_path = "../RedLights2011_Medium/"
f = open(prediction_path)
boxes = json.load(f)

sucesses = ['RL-001.jpg', 'RL-052.jpg']
failures = ['RL-012.jpg', 'RL-047.jpg']
for s in sucesses: 
    jpg = Image.open(data_path + s)
    draw_boxes(jpg, boxes[s])
    jpg.save(save_path + s)

for f in failures: 
    jpg = Image.open(data_path + f)
    draw_boxes(jpg, boxes[f])
    jpg.save(save_path + f)



