import re
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO

import torch
from torch import nn
from torchvision.transforms import Resize

from helpers import transform_sketch, calculate_and_save_accumulated_optical_flow, generate_tracking_img, generate_sketch_image

from rdp import rdp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tdigest import TDigest
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import warnings
from datasets import Dataset, load_dataset, load_from_disk

warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()


def read_ndjson(file_path, num_lines):
    with open(file_path, 'r') as file:
        data = []
        for i in range(num_lines):
            line = file.readline()
            # If line is empty, this means end of file is reached
            if not line:
                break
            data.append(pd.read_json(StringIO(line), typ='series'))
    # Convert the list of Series to a DataFrame
    return pd.DataFrame(data)



def preprocess_videos(video_path, frame_num, transform, start_timestamp=None, end_timestamp=None):
    vr = VideoReader(video_path)
    if start_timestamp is not None and end_timestamp is not None:
        frame_rate = vr.get_avg_fps()
        
        start_frame = int(start_timestamp * frame_rate)
        end_frame = min(len(vr)-1, int(end_timestamp * frame_rate))
        total_frames = end_frame - start_frame
    
        if total_frames > frame_num:
            step_size = np.linspace(start_frame, end_frame, num=frame_num, endpoint=True, dtype=int)
        else:
            step_size = np.arange(start_frame, end_frame, dtype=int)
        
        video = torch.tensor(vr.get_batch(step_size).asnumpy())
    else:
        frame_num = min(frame_num, len(vr))
        frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
        video = torch.tensor(vr.get_batch(frame_idx).asnumpy())
    video = video.permute(-1,0,1,2).float() / 255.
    video = video[:,:,52:52+1028, 37:37+1028].clone()
    video = transform(video)
    return video


def preprocess_sketch(sketch_data, do_rdp=False, rdp_epsilon=0.02, dataset='quickdraw'):
    transformed_strokes = transform_sketch(sketch_data, dataset=dataset)
    xyt_strokes = []
    for stroke in transformed_strokes:
        xyt_pairs = [(x,y,t) for x,y,t in zip(*stroke)]
        if do_rdp:
            xyt_pairs = rdp(xyt_pairs, rdp_epsilon)

        xyt_strokes.append(xyt_pairs)
    return xyt_strokes


def preprocess(video_root, sketch_ndjson_root_path, out_dir, size, frame_num, do_rdp=False, rdp_epsilon=None):
    video_transform = Resize(size)
    os.makedirs(out_dir, exist_ok=True)
    
    for root, dir, file in os.walk(video_root):
        
        if root == video_root: continue

        category = root.split('/')[-1]
        if os.path.exists(os.path.join(out_dir, category)):
            print(f'{category} already created')
            continue

        df = read_ndjson(os.path.join(sketch_ndjson_root_path, f'{category}.ndjson'), 105)['drawing']
        for f in tqdm(file, desc=category):
            if f.endswith('orig.mp4'):
                video_filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                save_file_path = os.path.join(out_dir, category, f'{sketch_id}.pt')
                if os.path.exists(save_file_path):
                    print(f'{category}/{sketch_id} already created')
                    continue
                drawing = df[int(sketch_id)]
                video = preprocess_videos(video_filename, frame_num, video_transform)
                sketch = preprocess_sketch(drawing, do_rdp, rdp_epsilon)
                os.makedirs(os.path.join(out_dir, category), exist_ok=True)
                torch.save({'video_filename': video_filename, 
                            'category': category, 
                            'sketch_id': sketch_id, 
                            'video': video, 
                            'sketch': sketch}, save_file_path)


def preprocess_sketches(sketch_ndjson_root_path, out_file, do_rdp=False, rdp_epsilon=None):

    result = {}
    
    for f in os.listdir(sketch_ndjson_root_path):

        category = f.split('.')[0]
        result[category] = {}

        df = read_ndjson(os.path.join(sketch_ndjson_root_path, f), 105)['drawing']
        for i in tqdm(range(len(df)), desc=category):
            sketch = preprocess_sketch(df[i], do_rdp, rdp_epsilon)
            result[category][i] = sketch
    
    json.dump(result, open(out_file, 'w'))
                

def add_idx_to_ndjsons(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in tqdm(os.listdir(root)):

        df = pd.read_json(os.path.join(root, f), lines=True)

        df.reset_index(inplace=True)

        df.to_json(os.path.join(out_dir, f), orient='records', lines=True)


def preprocess_sketches_as_hf_dataset(sketch_ndjson_root_path, out_dir):

    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, device_map = 'cuda')
    # model.eval()
    image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, device_map = 'cuda')
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, device_map = 'cuda')
    image_model.eval()
    text_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16, device_map = 'cuda')

    class AesthScoreModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

    aespredictor = AesthScoreModel(768)
    aespredictor.load_state_dict(torch.load('/home/scui/projects/DoodleFusion/ckpts/sac+logos+ava1-l14-linearMSE.pth'))
    aespredictor.eval()
    aespredictor = aespredictor.to('cuda').half()

    num_proc = os.cpu_count()
    pattern = sketch_ndjson_root_path + '*.ndjson'  
    dataset = load_dataset('json', data_files=pattern, num_proc=num_proc)
    stats_per_category = {}
    def map_func(examples):
        categories = []
        drawings = []
        images = []
        indices = []
        for category, is_recognized, indice, data in zip(examples['word'], examples['recognized'], examples['index'], examples['drawing']):
            if is_recognized:
                categories.append(category)
                # if category not in stats_per_category:
                #     stats_per_category[category] = {'count': 0}
                # stats_per_category[category]['count'] += 1
                preprocessed_data = preprocess_sketch(data)
                drawings.append(preprocessed_data)
                images.append(generate_sketch_image(preprocessed_data, (224,224)))
                indices.append(indice)

        inputs = processor(text=[f"a sketch drawing of a {category}" for category in categories], images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to('cuda') for k,v in inputs.items()}

        with torch.no_grad():
            # outputs = model(**inputs)
            # logits_per_image = outputs.logits_per_image.diag().cpu() / 100
            image_features = image_model(inputs['pixel_values']).image_embeds
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = text_model(inputs['input_ids']).text_embeds
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (image_features @ text_features.T).diag().cpu().tolist()
            aesthetic_score = aespredictor(image_features).squeeze(-1).cpu().tolist()
        
        for category, clip_score, aes_score in zip(categories, logits_per_image, aesthetic_score):
            if category not in stats_per_category:
                stats_per_category[category] = {'count': 0, 'sum': 0, 'aes_sum':0, 'percentile': TDigest(), 'aes_percentile': TDigest()}
            stats_per_category[category]['count'] += 1
            stats_per_category[category]['sum'] += clip_score
            stats_per_category[category]['aes_sum'] += aes_score
            stats_per_category[category]['percentile'].update(clip_score)
            stats_per_category[category]['aes_percentile'].update(aes_score)

        logits_per_image = [round(logit, 4) for logit in logits_per_image]
        aesthetic_score = [round(aes, 4) for aes in aesthetic_score]

        return {'category': categories, 'drawing': drawings, 'id': indices, 'clip_score': logits_per_image, "aes_score": aesthetic_score}

    dataset = dataset.map(map_func, remove_columns=['countrycode', 'timestamp', 'key_id', 'recognized', 'drawing', 'word', 'index'], batched=True, num_proc=None)
    dataset.save_to_disk(out_dir)

    # categories = []
    # for f in os.listdir(sketch_ndjson_root_path):
    #     cat = f.split('.')[0]
    #     categories.append(cat)

    # categories.sort()

    for k,v in stats_per_category.items():
        stats_per_category[k]['mean'] = v['sum'] / v['count']
        stats_per_category[k]['aes_mean'] = v['aes_sum'] / v['count']
        del stats_per_category[k]['sum']
        del stats_per_category[k]['aes_sum']
        p_25, p_50, p_75, p_95 = v['percentile'].percentile(25), v['percentile'].percentile(50), v['percentile'].percentile(75), v['percentile'].percentile(95)
        del stats_per_category[k]['percentile']
        stats_per_category[k]['percentile'] = {'25': p_25, '50': p_50, '75': p_75, '95': p_95}

        p_25, p_50, p_75, p_95 = v['aes_percentile'].percentile(25), v['aes_percentile'].percentile(50), v['aes_percentile'].percentile(75), v['aes_percentile'].percentile(95)
        del stats_per_category[k]['aes_percentile']
        stats_per_category[k]['aes_percentile'] = {'25': p_25, '50': p_50, '75': p_75, '95': p_95}

    with open(os.path.join(out_dir, 'category_stats.json'), 'w') as f:
        json.dump(stats_per_category, f)

def preprocess_first_stroke(video_root, sketch_ndjson_root_path, out_dir, size, frame_num, do_rdp=False, rdp_epsilon=None):
    video_transform = Resize(size)
    os.makedirs(out_dir, exist_ok=True)
    
    for root, dir, file in os.walk(video_root):
        
        if root == video_root: continue

        category = root.split('/')[-1]
        if os.path.exists(os.path.join(out_dir, category)):
            print(f'{category} already created')
            continue

        df = read_ndjson(os.path.join(sketch_ndjson_root_path, f'{category}.ndjson'), 105)['drawing']
        timestamp_json = json.load(open(os.path.join(root, f"{category}Timestamps.json")))['timestamps']
        for f in tqdm(file, desc=category):
            if f.endswith('orig.mp4'):
                video_filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                save_file_path = os.path.join(out_dir, category, f'{sketch_id}.pt')
                # if os.path.exists(save_file_path):
                #     print(f'{category}/{sketch_id} already created')
                #     continue
                drawing = df[int(sketch_id)]
                first_stroke_start_time, first_stroke_end_time = timestamp_json[int(sketch_id)][0]
                if first_stroke_end_time - first_stroke_start_time < 2:
                    continue
                video = preprocess_videos(video_filename, frame_num, video_transform, first_stroke_start_time, first_stroke_end_time)
                sketch = preprocess_sketch(drawing, do_rdp, rdp_epsilon)[:1]
                os.makedirs(os.path.join(out_dir, category), exist_ok=True)
                torch.save({'video_filename': video_filename, 
                            'category': category, 
                            'sketch_id': sketch_id, 
                            'video': video, 
                            'sketch': sketch}, save_file_path)



def partition_video_by_stroke(video_root, sketch, has_finger_movement, transform=None, fps=60, frame_per_stroke=16):
    # first get the video and time intervals as frame num for each stroke
    # for each stroke uniformly sample 16 frames
    vr = VideoReader(video_root)
    frame_num = len(vr)
    last_frame_idx = frame_num - 1
    stroke_intervals = []
    # sketch = sketch[:-1]
    # (MS_total / 1000 * fps) * (1+lag) + (n_strokes-1) * 2 * fps = frame_num
    if has_finger_movement:
        lag = (frame_num - (len(sketch)-1) * 2 * fps) / (sketch[-1][-1][-1]/1000 * fps)
    else:
        lag = frame_num / (sketch[-1][-1][-1]/1000 * fps)

    for i, stroke in enumerate(sketch):
        if has_finger_movement:
            current_stroke_start, current_stroke_end = int(stroke[0][-1]/1000 * fps * lag + i * 2 * fps) , int(stroke[-1][-1]/1000 * fps * lag + i * 2 * fps)
            if i < len(sketch) -1:
                # account for the added 2 secs for finger movement and the time between strokes
                current_stroke_end = int(sketch[i+1][0][-1]/1000 * fps * lag + (i+1) * 2 * fps)
                if current_stroke_end > last_frame_idx:
                    current_stroke_end = last_frame_idx
                    break
            else:
                if current_stroke_end != last_frame_idx:
                    current_stroke_end = last_frame_idx
        else:
            current_stroke_start, current_stroke_end = int(stroke[0][-1]/1000 * fps * lag), int(stroke[-1][-1]/1000 * fps * lag)
            if i < len(sketch) -1:
                current_stroke_end = int(sketch[i+1][0][-1]/1000 * fps * lag)
                if current_stroke_end > last_frame_idx:
                    current_stroke_end = last_frame_idx
                    break
            else:
                if current_stroke_end != last_frame_idx:
                    current_stroke_end = last_frame_idx
        sample_idx = np.linspace(current_stroke_start, current_stroke_end, frame_per_stroke).astype(int).tolist()
        stroke_intervals += sample_idx
        # for sanity check
        if i > 2:
            break
    
    sampled_frames = torch.tensor(vr.get_batch(stroke_intervals).asnumpy())
    sampled_strokes = [sampled_frames[i*frame_per_stroke:(i+1)*frame_per_stroke] for i in range(len(stroke_intervals)//frame_per_stroke)]
    sampled_strokes = []

    # for i in range(len(stroke_intervals)//frame_per_stroke):
    for i in range(min(2, len(stroke_intervals)//frame_per_stroke)):
        video = sampled_frames[i*frame_per_stroke:(i+1)*frame_per_stroke]

        video = video.permute(-1,0,1,2).float() / 255.
        video = video[:,:,52:52+1028, 37:37+1028].clone()
        video = transform(video) if transform else video
        sampled_strokes.append(video)
        
    
    return sampled_strokes


def preprocess_with_separate_strokes(video_root, sketch_ndjson_root_path, out_dir, size, frame_num, do_rdp=False, rdp_epsilon=None):
    video_transform = Resize(size)
    os.makedirs(out_dir, exist_ok=True)
    
    for root, dir, file in os.walk(video_root):
        
        if root == video_root: continue

        category = root.split('/')[-1]
        if os.path.exists(os.path.join(out_dir, category)):
            print(f'{category} already created')
            continue

        df = pd.read_json(os.path.join(sketch_ndjson_root_path, f'raw_{category}.ndjson'), lines=True)['drawing']
        os.makedirs(os.path.join(out_dir, category), exist_ok=True)

        for f in tqdm(file, desc=category):
            if f.endswith('orig.mp4'):
                video_filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                sketch = preprocess_sketch(df[int(sketch_id)], size, do_rdp, rdp_epsilon)

                partitioned_videos = partition_video_by_stroke(video_filename, sketch, True, video_transform) 

                for i in range(len(partitioned_videos)):
                    
                    torch.save({'video_filename': video_filename, 
                                'category': category, 
                                'sketch_id': sketch_id, 
                                'video': partitioned_videos[i], 
                                'sketch': [sketch[i]]}, os.path.join(out_dir, category, f'{sketch_id}_{i}.pt'))


import matplotlib.pyplot as plt
def visualize_sketch_raw(sketch_data, save_path=None):
    fig, ax = plt.subplots()
    for stroke in sketch_data:
        ax.plot(stroke[0], stroke[1], marker='o')
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # Invert y-axis to match the image coordinate system
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_sketch(sketch_data, save_path=None):
    """
        for processed dataformat: [stroke1, stroke2, ...], stroke1 = [[x1, y1], [x2, y2], ...]
    """
    fig, ax = plt.subplots()
    for stroke in sketch_data:
        x = [i[0] for i in stroke]
        y = [i[1] for i in stroke]
        ax.plot(x, y, marker='o')
    ax.set_aspect('equal', 'box')
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.invert_yaxis()  # Invert the y-axis to match the drawing coordinate system
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_flow(video_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    for root, dir, file in os.walk(video_root):
        
        if root == video_root: continue

        category = root.split('/')[-1]
        if os.path.exists(os.path.join(out_dir, category)):
            print(f'{category} already created')
            continue
        
        os.makedirs(os.path.join(out_dir, category), exist_ok=True)
        for f in tqdm(file, desc=category):
            if f.endswith('orig.mp4'):
                video_filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                save_file_path = os.path.join(out_dir, category, f'{sketch_id}.png')
                if os.path.exists(save_file_path):
                    print(f'{save_file_path} already created')
                    continue
                
                calculate_and_save_accumulated_optical_flow(video_filename, save_file_path)


def generate_mp_tracking(video_root, image_out_dir, coord_out_path):
    os.makedirs(image_out_dir, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    
    trajectories = {}
    for root, dir, file in os.walk(video_root):
        
        if root == video_root: continue

        category = root.split('/')[-1]
        if os.path.exists(os.path.join(image_out_dir, category)):
            print(f'{category} already created')
            continue
        
        trajectories[category] = {}
        os.makedirs(os.path.join(image_out_dir, category), exist_ok=True)
        for f in tqdm(file, desc=category):
            if f.endswith('.MOV') or f.endswith('.mp4'):
                video_filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                save_file_path = os.path.join(image_out_dir, category, f'{sketch_id}.png')
                if os.path.exists(save_file_path):
                    print(f'{save_file_path} already created')
                    continue
                
                trajectory, _ = generate_tracking_img(video_filename, save_file_path, line_width=2, detector=detector)
                trajectories[category][sketch_id] = trajectory
    json.dump(trajectories, open(coord_out_path, 'w'))
                

def filter_hf_dataset(path, save_path):

    held_out_categories = {"cat", "car", "face", "snail", "sun", "candle", "angel", "grapes", "cow", "diamond"}
    category_stats = json.load(open(os.path.join(path, 'category_stats.json'), 'r'))
    dataset = load_from_disk(path)

    def filter_func(example):
    
        if int(example['id']) < 100:
            return False

        if held_out_categories and example['category'] in held_out_categories:
            return False

        threshold = category_stats[example['category']]['percentile'][str(95)]
        if example['clip_score'] < threshold:
            return False

        return True

    dataset = dataset.filter(filter_func, num_proc=os.cpu_count())

    dataset.save_to_disk(save_path, num_proc=os.cpu_count())
    json.dump(category_stats, open(os.path.join(save_path, "category_stats.json"), 'w'))

            
import xml.etree.ElementTree as ET
import numpy as np

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def cubic_bezier(p0, p1, p2, p3, t):
    """Calculate the point on the cubic Bézier curve at parameter t."""
    # B(t) = (1-t)^3 * P0 + 3 * (1-t)^2 * t * P1 + 3 * (1-t) * t^2 * P2 + t^3 * P3
    return ((1-t)**3 * np.array(p0) +
            3 * (1-t)**2 * t * np.array(p1) +
            3 * (1-t) * t**2 * np.array(p2) +
            t**3 * np.array(p3))

def parse_svg_path_data(path_data, point_density=10):
    """Parses the 'd' attribute of an SVG path element to extract points and handles cubic Bézier curves."""
    commands = path_data.replace(',', ' ').split()
    parsed_commands = []
    for cmd in commands:
        if 'NaN' in cmd:
            return None
        if  'M' in cmd[1:] or 'L' in cmd[1:] or 'C' in cmd[1:]:
            idx = cmd[1:].index('M') if 'M' in cmd[1:] else cmd[1:].index('L') if 'L' in cmd[1:] else cmd[1:].index('C')
            idx += 1
            parsed_commands.append(cmd[:idx])
            parsed_commands.append(cmd[idx:])
        else:
            parsed_commands.append(cmd)
    commands = parsed_commands

    strokes = []
    stroke = []
    i = 0
    while i < len(commands):
        cmd = commands[i]
        if cmd.startswith('M'):
            if stroke:  # if there's any existing stroke, append it to strokes
                strokes.append(np.array(stroke)/400)
                stroke = []
            x, y = float(cmd[1:]), float(commands[i + 1])
            stroke.append((x, y))
            i += 2
        elif cmd.startswith('L'):
            x, y = float(cmd[1:]), float(commands[i + 1])
            stroke.append((x, y))
            i += 2
        elif cmd.startswith('C'):
            x0, y0 = stroke[-1]
            x1, y1 = float(cmd[1:]), float(commands[i + 1])
            x2, y2 = float(commands[i + 2]), float(commands[i + 3])
            x3, y3 = float(commands[i + 4]), float(commands[i + 5])
            length = (distance((x1, y1), (x2, y2)) + 
                      distance((x2, y2), (x3, y3)))
            for t in np.linspace(0, 1, int(length)):
                pt = cubic_bezier((x0, y0), (x1, y1), (x2, y2), (x3, y3), t)
                stroke.append((pt[0], pt[1]))
            i += 6
        else:
            raise Exception(f'Unrecognized command: {cmd}')
            i += 1  # Skip unrecognized commands
    if stroke:
        strokes.append(np.array(stroke)/400)
    return strokes

def svg_to_strokes(file_path):
    """Converts an SVG file into a list of strokes, with each stroke being a list of (x, y) coordinates."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    strokes = []
    
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        path_data = path.get('d')
        parsed_data = parse_svg_path_data(path_data)
        if parsed_data is None:
            print(f'NaN encountered in {file_path}, skipping')
            continue
        else:
            strokes.extend(parsed_data)
    
    return strokes

def tuberlin_svg_to_array(svg_root, save_path):

    drawings = []
    categories = []
    idx = []

    for root, _, file in os.walk(svg_root):
        
        if root == svg_root: continue

        category = root.split('/')[-1]

        for count, f in enumerate(tqdm(file, desc=category)):
            if f.endswith('.svg'):
                filename = os.path.join(root, f)
                sketch_id = re.search(r"\d+", f).group(0)
                sketch = svg_to_strokes(filename)
                sketch = transform_sketch(sketch, dataset='tuberlin')
                drawings.append(sketch)
                categories.append(category)
                idx.append(str(count))

    dataset = Dataset.from_dict({"drawing": drawings, "category": categories, "id": idx})
    dataset.save_to_disk(os.path.join(save_path, 'preprocessed_hf_dataset'), num_proc=os.cpu_count())

    dict_json = {}
    for drawing, category, idx in zip(drawings, categories, idx):
        if category not in dict_json:
            dict_json[category] = {}
        dict_json[category][idx] = drawing

    json.dump(dict_json, open(os.path.join(save_path, 'sketches.json'), 'w'))


# start_of_stroke, end_of_stroke, transition+2s finger-movement

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for qd dataset
    parser.add_argument("--quickdraw", action="store_true", help="Set if preprocessing the Quick, Draw! dataset.")
    parser.add_argument("--quickdraw_ndjsons", type=str, default="datasets/quickdraw_ndjsons_raw_all", help="Path to directory where ndjsons are saved.")
    # for hand motion video datasets
    parser.add_argument("--handtracking", action="store_true", help="Set if preprocessing hand motion video datasets.")
    parser.add_argument("--video_directory", type=str, help="Path to directory where hand motion videos are saved.")
    parser.add_argument("--dataset_type", choices=["real", "synthetic"], default="real", help="Which hand motion dataset (real or synthetic) is being processed."
    )
    args = parser.parse_args()

    if args.quickdraw:
        # create hf dataset from quickdraw data
        indexed_dir = os.path.join(os.path.dirname(args.quickdraw_ndjsons), "quickdraw_ndjsons_indexed")
        add_idx_to_ndjsons(root=args.quickdraw_ndjsons, outdir=indexed_dir)
        full_dataset_dir = os.path.join(os.path.dirname(args.quickdraw_ndjsons), "preprocessed_quickdraw_hf_dataset") 
        preprocess_sketches_as_hf_dataset(indexed_dir, full_dataset_dir)
        filtered_dataset_dir = os.path.join(os.path.dirname(args.quickdraw_ndjsons), "preprocessed_filtered_quickdraw_hf_dataset") 
        filter_hf_dataset(full_dataset_dir, filtered_dataset_dir)
        
        # also consolidate all the ndjsons into one single json
        sketch_json = os.path.join(os.path.dirname(args.quickdraw_ndjsons), "sketches_full.json") 
        preprocess_sketches(args.quickdraw_ndjsons, sketch_json)
        
    if args.handtracking:
        image_directory = os.path.join(os.path.dirname(args.video_directory), f"{args.dataset_type}_handtracked_sketches")
        coords_directory = os.path.join(os.path.dirname(args.video_directory), f"{args.dataset_type}_handtracked_sketch_coordinates")
        generate_mp_tracking(args.video_directory, image_directory, coords_directory)