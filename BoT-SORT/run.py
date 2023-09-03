import json
import re
import sys
import time
import requests
import gdown

import torch
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, scale_coords, non_max_suppression
from yolov7.utils.torch_utils import select_device, time_synchronized

from tracker.mc_bot_sort import BoTSORT

sys.path.insert(0, './yolov7/')
sys.path.append('.')

sample_video_data = None

def init():
    #file_id = "1KyRJNgfApv3m7cHdW7Ekt87pxrs_3ozu"
    #gdown.download(f'https://drive.google.com/uc?id={file_id}', "./prbnet.pt", quiet=False)
    print('Start downloading yolov7 pretrained weights.')
    
    url = f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    response = requests.get(url)
    if response.status_code == 200:
        with open('./yolov7.pt', "wb") as f:
            f.write(response.content)
            print('successfully downloaded yolov7 weights')
    else:
        print('failed downloading yolov7 weights.')

    

def load_sample_videos():
    json_file_path = './data/sample/sample-captions.json'
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        
    global sample_video_data 
    sample_video_data = list(json_data)

def get_sample_video(index):
    global sample_video_data 
    sample = sample_video_data[index]
    
    video_id = sample['video_id']
    video_file_path = f'./data/sample/{video_id}.mov'
    desc = sample['desc']
    return video_file_path, desc
    
def extract_timeframes(input_string):
    pattern = r'\d{2}:(\d{2})-\d{2}:(\d{2})'
    timeframes = re.findall(pattern, input_string)
    timeframes = [(int(sec1), int(sec2)) for (sec1, sec2) in timeframes]
    
    timely_descs = input_string.split('\n')[:-1]
    assert len(timeframes) == len(timely_descs)

    return timeframes, timely_descs

def detect(source):
    weights = './yolov7.pt'
    imgsz = 1920
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    min_box_area = 2000

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    class Args:
        def __init__(self):
            self.track_high_thresh = 0.3
            self.track_low_thresh = 0.05
            self.new_track_thresh = 0.4
            self.track_buffer = 30
            self.proximity_thresh = 0.5
            self.appearance_thresh = 0.25
            self.cmc_method = "sparseOptFlow"
            self.name = "exp"
            self.ablation = None
            self.with_reid = False
            self.mot20 = False
            self.match_thresh = 0.7
            self.aspect_ratio_thresh = 1.6
    args = Args()
    tracker = BoTSORT(args, frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    f_results = []
    frame_num = 0
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        frame_num += 1

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        conf_thres, iou_thres = 0.09, 0.7
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.detach().numpy()
                detections = det.detach().numpy()
                detections[:, :4] = boxes

            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    f_results.append(
                        {
                          "frame-id": frame_num,
                          "target-id": tid,
                          "x": tlwh[0],
                          "y": tlwh[1],
                          "w": tlwh[2],
                          "h": tlwh[3],
                          "label": names[int(tcls)]
                        }
                    )
    return f_results

def main():
    #init()
    load_sample_videos()
    for i in range(0, 3):
        video_file_path, desc = get_sample_video(i)
        timeframes, timely_descs = extract_timeframes(desc)

        mot_result = detect(video_file_path)
        print(mot_result)

        # 30 * timeframe에 해당하는 object들 모아서 description 완성 후
        # timely_descs와 interleave하게 이어붙일 것.

        break
        

if __name__=='__main__':
    main()