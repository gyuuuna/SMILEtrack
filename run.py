import json
import re

sample_video_data = None

def load_sample_videos():
    json_file_path = 'data/sample/sample-captions.json'
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        
    global sample_video_data 
    sample_video_data = list(json_data)

def get_sample_video(index):
    global sample_video_data 
    sample = sample_video_data[index]
    
    video_id = sample['video_id']
    video_file_path = f'data/sample/{video_id}.mov'
    desc = sample['desc']
    return video_file_path, desc
    
def extract_timeframes(input_string):
    pattern = r'\d{2}:(\d{2})-\d{2}:(\d{2})'
    timeframes = re.findall(pattern, input_string)
    timeframes = [(int(sec1), int(sec2)) for (sec1, sec2) in timeframes]
    
    timely_descs = input_string.split('\n')[:-1]
    assert len(timeframes) == len(timely_descs)

    return timeframes, timely_descs

def main():
    load_sample_videos()
    for i in range(0, 3):
        _, desc = get_sample_video(i)
        extract_timeframes(desc)

if __name__=='__main__':
    main()