import matplotlib.pyplot as plt
import torch
from decord import VideoReader
import random
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import trace_skeleton
from rdp import rdp
import cv2
from skimage.morphology import skeletonize
import xml.etree.ElementTree as ET


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


def preprocess_videos_numpy(video_path, frame_num, start_timestamp=None, end_timestamp=None):
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
        
        video = vr.get_batch(step_size).asnumpy()
    else:
        frame_num = min(frame_num, len(vr))
        frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
        video = vr.get_batch(frame_idx).asnumpy()
    video = np.transpose(video, (-1,0,1,2)).astype(np.float32) / 255.
    video = video[:,:,52:52+1028, 37:37+1028].copy()
    return video


class RandomResizePadHflip:
    def __init__(self, orig_size=(1024,1024), do_hflip=True, min_scale=0.5, max_scale=1):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.original_height, self.original_width = orig_size
        self.do_hflip = do_hflip

    def random_initialize(self):
        self.scale_factor_x = random.uniform(self.min_scale, self.max_scale)
        self.scale_factor_y = random.uniform(self.min_scale, self.max_scale)
        if self.do_hflip:
            self.do_hflip = random.random() > 0.5
        self.new_height = int(self.original_height * self.scale_factor_y)
        self.new_width = int(self.original_width * self.scale_factor_x)
        pad_height = self.original_height - self.new_height
        pad_width = self.original_width - self.new_width
        self.pad_top = random.randint(0, pad_height)
        self.pad_left = random.randint(0, pad_width)
        self.pad_right = pad_width - self.pad_left
        self.pad_bottom = pad_height - self.pad_top

        self.pad_top_sketch = self.pad_top / self.original_height
        self.pad_left_sketch = self.pad_left / self.original_width


    def transform_video(self, video_tensor):

        # resized_image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(self.new_height, self.new_width), mode='bilinear', align_corners=False).squeeze(0)
        resized_video_tensor = transforms.Resize((self.new_height, self.new_width))(video_tensor)
        final_video_tensor = F.pad(resized_video_tensor, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), 'constant', 1)
        if self.do_hflip:
            final_video_tensor = hflip(final_video_tensor)
        return final_video_tensor

    def transform_flow(self, flow_tensor):

        # resized_image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(self.new_height, self.new_width), mode='bilinear', align_corners=False).squeeze(0)
        resized_flow_tensor = transforms.Resize((self.new_height, self.new_width))(flow_tensor)
        final_flow_tensor = F.pad(resized_flow_tensor, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), 'constant', 0)
        if self.do_hflip:
            final_flow_tensor = hflip(final_flow_tensor)
        return final_flow_tensor
    
    def transform_sketch(self, sketch_data):
        resized_sketch_data = [[[x*self.scale_factor_x, y*self.scale_factor_y, t] for x, y, t in stroke] for stroke in sketch_data]
        padded_sketch_data = [[(x + self.pad_left_sketch, y + self.pad_top_sketch, t) for x, y, t in stroke] for stroke in resized_sketch_data]

        if self.do_hflip:
            flipped_sketch_data = [[(1 - x, y, t) for x, y, t in stroke] for stroke in padded_sketch_data]
            return flipped_sketch_data
        else:
            return padded_sketch_data


import torch.nn.functional as F

def normalize_sketch(sketch):

    if sketch.dim() != 3 or sketch.size(0) != 3:
        raise ValueError("Input tensor must have shape (3, N, N)")

    non_background_indices = (sketch == 1).nonzero(as_tuple=True)
    min_x, max_x = non_background_indices[1].min(), non_background_indices[1].max()
    min_y, max_y = non_background_indices[2].min(), non_background_indices[2].max()

    cropped_sketch = sketch[:, min_x:max_x+1, min_y:max_y+1]

    return cropped_sketch


def calculate_slope_distance(points, slope_mapping="polar", normalize_angle=True):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float)
    
    deltas = points[1:] - points[:-1]
    
    angles = torch.atan2(deltas[:, 1], deltas[:, 0])

    if normalize_angle:
        angles = (angles + torch.pi) / (2 * torch.pi)
    
    distances = torch.sqrt(torch.sum(deltas[:,:-1] ** 2, axis=1))
    
    return torch.stack([angles, distances, deltas[:,2]], axis=1)


def calculate_slope_distance_for_multi_strokes(sketch, slope_mapping="polar", normalize_angle=True):
    lengths_per_stroke = [len(stroke) for stroke in sketch]
    if lengths_per_stroke[0] > 1:
        lengths_per_stroke[0] -= 1
    elif lengths_per_stroke[-1] > 1:
        lengths_per_stroke[-1] -= 1
    else:
        lengths_per_stroke = lengths_per_stroke[:-1]
    sketch = torch.cat([torch.tensor(stroke, dtype=torch.float) for stroke in sketch])
    slope_distance = calculate_slope_distance(sketch, slope_mapping, normalize_angle)
    return torch.split(slope_distance, lengths_per_stroke, dim=0)


def visualize_sketch(sketch_data):
    fig, ax = plt.subplots()
    for stroke in sketch_data:
        ax.plot(stroke[0], stroke[1], marker='o')
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # Invert y-axis to match the image coordinate system
    plt.show()

def visualize_sketch_raw(sketch_data):
    fig, ax = plt.subplots()
    for stroke in sketch_data:
        ax.plot(stroke[0], stroke[1], marker='o')
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # Invert y-axis to match the image coordinate system
    plt.show()


def pad_list_of_lists(list_of_lists, padding_value=torch.tensor([-100,-100, -100]), padding_side='left', pad_to_length=None, truncation=None):

    pad_to_length = pad_to_length if pad_to_length is not None else max(len(inner_list) for inner_list in list_of_lists)
    pad_to_length = min(pad_to_length, truncation) if truncation is not None else pad_to_length
    attention_masks = []
    padded_lists = []
    for inner_list in list_of_lists:
        padding_len = pad_to_length - len(inner_list)
        if padding_len < 0:
            padded_lists.append(torch.tensor(inner_list[:pad_to_length]))
            attention_mask = [1] * pad_to_length
            attention_masks.append(torch.tensor(attention_mask))
            continue
        if padding_side == 'left':
            padded_lists.append(torch.tensor([padding_value] * padding_len + inner_list))
            attention_mask = [0] * padding_len + [1] * len(inner_list)
        else:
            padded_lists.append(torch.tensor(inner_list + [padding_value] * padding_len))
            attention_mask = [1] * len(inner_list) + [0] * padding_len
        attention_masks.append(torch.tensor(attention_mask))
    
    return padded_lists, attention_masks

def generate_sketch(sketch_data, resolution=(1024, 1024), background_color=(255, 255, 255), line_color=(0, 0, 0), line_width=2):
    """
    Generate a sketch picture from a list of strokes.

    :param sketch_data: List of strokes, where each stroke is a list of (x, y) tuples.
    :param resolution: Tuple indicating the image resolution (width, height).
    :param background_color: Tuple indicating the RGB background color of the image.
    :param line_color: Tuple indicating the RGB color of the sketch lines.
    :param line_width: Thickness of the sketch lines.
    :return: A PIL Image object containing the generated sketch.
    """
    # Create a new image with the specified resolution and background color
    image = Image.new('RGB', resolution, background_color)
    draw = ImageDraw.Draw(image)
    
    # Iterate through each stroke and draw lines between consecutive points
    for stroke in sketch_data:
        if len(stroke) > 1:  # Ensure the stroke has at least two points to draw a line
            draw.line(stroke, fill=line_color, width=line_width)
    
    return image



def transform_sketch(sketch, normalize_t=False, dataset='quickdraw'):

    """
        Re-position sketch to top-left corner and reshape
        Normalize x,y, and t to [0,1]
    """

    if dataset == 'quickdraw':
        # if sketch is in the form [[x1, y1, t1], [x2, y2, t2] ...] x,y,t are arrays
        all_x = [x for stroke in sketch for x in stroke[0]]
        all_y = [y for stroke in sketch for y in stroke[1]]
    elif dataset == 'tuberlin':
        all_x = [point[0] for stroke in sketch for point in stroke]
        all_y = [point[1] for stroke in sketch for point in stroke]
    # all_t = [t for stroke in sketch for t in stroke[2]]

    min_x, min_y = min(all_x), min(all_y)
    max_x, max_y = max(all_x), max(all_y)
    # min_t, max_t = min(all_t), max(all_t)

    current_width = max_x - min_x
    current_height = max_y - min_y
    # total_t = max_t - min_t

    # scale_x = new_width / current_width
    # scale_y = new_height / current_height
    # scale_factor = min(scale_x, scale_y) # if we want to keep the aspect ratio, but the video is square

    norm_factor_x = current_width if current_width > 0 else 1
    norm_factor_y = current_height if current_height > 0 else 1

    transformed_sketch = []
    for stroke in sketch:
        if dataset == 'quickdraw':
            transformed_stroke_x = [(x - min_x) / norm_factor_x for x in stroke[0]]
            transformed_stroke_y = [(y - min_y) / norm_factor_y for y in stroke[1]]
            transformed_sketch.append([transformed_stroke_x, transformed_stroke_y])
        elif dataset == 'tuberlin':
            transformed_stroke = [[(point[0] - min_x) / norm_factor_x, (point[1] - min_y) / norm_factor_y] for point in stroke]
            transformed_sketch.append(transformed_stroke)

    return transformed_sketch


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
    ax.invert_yaxis()  # Invert the y-axis to match the drawing coordinate system
    plt.xlim(0,1)
    plt.ylim(0,1)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

import pandas as pd
from io import StringIO

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


from PIL import Image, ImageDraw
def generate_sketch_image(sketch_data, resolution=(256, 256), background_color=(255, 255, 255), line_color=(0, 0, 0), line_width=2):

    image = Image.new('RGB', resolution, background_color)
    draw = ImageDraw.Draw(image)
    
    width, height = resolution
    
    scaled_sketch_data = [
        [(point[0] * width, point[1] * height) for point in stroke] for stroke in sketch_data
    ]
    
    for stroke in scaled_sketch_data:
        if len(stroke) > 1:  # Ensure the stroke has at least two points to draw a line
            draw.line(stroke, fill=line_color, width=line_width)
    
    return image


import cv2
def calculate_and_save_accumulated_optical_flow(video_path, output_image_path=None, draw_line_width=2):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and st is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), draw_line_width)
        
        old_gray = frame_gray.copy()
        if p1 is not None and st is not None:
            p0 = good_new.reshape(-1, 1, 2)

    if output_image_path:
        cv2.imwrite(output_image_path, mask)
    cap.release()
    return mask


MARGIN = 10 # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
	hand_landmarks_list = detection_result.hand_landmarks	
	handedness_list = detection_result.handedness
	annotated_image = np.copy(rgb_image)
	
	# Loop through the detected hands to visualize.
	for idx in range(len(hand_landmarks_list)):
		hand_landmarks = hand_landmarks_list[idx]
		handedness = handedness_list[idx]

	# Draw the hand landmarks.
	hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
	hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
	solutions.drawing_utils.draw_landmarks(
		annotated_image,
		hand_landmarks_proto,
		solutions.hands.HAND_CONNECTIONS,
		solutions.drawing_styles.get_default_hand_landmarks_style(),
		solutions.drawing_styles.get_default_hand_connections_style())

	# Get the top left corner of the detected hand's bounding box.
	height, width, _ = annotated_image.shape
	x_coordinates = [landmark.x for landmark in hand_landmarks]
	y_coordinates = [landmark.y for landmark in hand_landmarks]
	text_x = int(min(x_coordinates) * width)
	text_y = int(min(y_coordinates) * height) - MARGIN

	# Draw handedness (left or right hand) on the image.
	cv2.putText(annotated_image, f"{handedness[0].category_name}",
		(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
		FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

	return annotated_image


def generate_tracking_img(video_path, out_path=None, background_color=(255, 255, 255), line_width=2, detector=None):
    # Create an HandLandmarker object.
    if detector is None:
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    # process each frame
    trajectory = []
    processed_frames = []
    frame_number = 0
    shape = None

    while cap.isOpened(): 
        success, frame = cap.read()
        if not success:
            break

        if shape is None:
            shape = frame.shape
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks in this frame
        detection_result = detector.detect(frame_rgb)

        if detection_result.hand_landmarks:
            result = detection_result.hand_landmarks[0]
            if result:
                trajectory.append([result[8].x, result[8].y])

    image = Image.new('RGB', (shape[1], shape[0]), background_color)
    draw = ImageDraw.Draw(image)
    scaled_trajectory = [(int(x * shape[1]), int(y * shape[0])) for x, y in trajectory]
    draw.line(scaled_trajectory, fill=(0, 0, 0), width=line_width)
    if out_path:
        image.save(out_path)
    return trajectory, image

        
def increase_line_width(images, kernel_size=3):
    """
    Increase the line width of sketches in a batch of images.
    
    Parameters:
    - images: a batch of images as a torch tensor of shape (B, 3, H, W).
    - kernel_size: size of the dilation kernel. Default is 3.
    - padding: padding applied to the images. Default is 1 to maintain image size.
    
    Returns:
    - dilated_images: images with increased line width.
    """
    # Assuming your images are binary with 0 for background and 1 for sketch,
    # and they are in a float tensor format.
    
    # Convert images to grayscale by averaging the channels if they aren't already
    if images.shape[1] == 3:
        images = images.mean(dim=1, keepdim=True)
    
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=images.dtype, device=images.device)
    dilated_images = F.conv2d(images, dilation_kernel, padding=kernel_size // 2)
    dilated_images = torch.clamp(dilated_images, 0, 1)
    dilated_images_expanded = dilated_images.expand(-1, 3, -1, -1)
    return dilated_images_expanded

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def merge_strokes(strokes, threshold):
    """Merge strokes based on a distance threshold, combines checking and merging."""
    for i, stroke1 in enumerate(strokes):
        for j in range(len(strokes)):
            if i != j:
                stroke2 = strokes[j]
                if euclidean_distance(stroke1[-1], stroke2[0]) < threshold:
                    strokes[i] = stroke1 + stroke2
                    del strokes[j]
                    return merge_strokes(strokes, threshold)
                elif euclidean_distance(stroke1[0], stroke2[-1]) < threshold:
                    strokes[i] = stroke2 + stroke1
                    del strokes[j]
                    return merge_strokes(strokes, threshold)
                elif euclidean_distance(stroke1[0], stroke2[0]) < threshold:
                    strokes[i] = stroke2[::-1] + stroke1
                    del strokes[j]
                    return merge_strokes(strokes, threshold)
                elif euclidean_distance(stroke1[-1], stroke2[-1]) < threshold:
                    strokes[i] = stroke1 + stroke2[::-1]
                    del strokes[j]
                    return merge_strokes(strokes, threshold)
    return strokes

def extract_coord_sequence_from_image(img, merge_threshold=10, rdp_epsilon=2):
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) < 128:
        img = 255 - img
    img = np.array(img.convert('L'))
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(binary_img, (5, 5), 0)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Convert to boolean array and apply skeletonization
    skeleton = skeletonize(binary_img)
    coord_sequence = trace_skeleton.from_numpy(skeleton)
    coord_sequence = merge_strokes([p for p in coord_sequence if len(p)>1], merge_threshold)
    coord_sequence = [rdp(p, rdp_epsilon) for p in coord_sequence]
    return coord_sequence

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

def parse_svg_path_data(path_data):
    """Parses the 'd' attribute of an SVG path element to extract points and handles cubic Bézier curves."""
    commands = path_data.replace(',', ' ').split()
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
            for t in np.linspace(0, 1, int(length/2)):
                pt = cubic_bezier((x0, y0), (x1, y1), (x2, y2), (x3, y3), t)
                stroke.append((pt[0], pt[1]))
            i += 6
        else:
            print('Unrecognized command:', cmd)
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
        strokes.extend(parse_svg_path_data(path_data))
    
    return strokes


if __name__ == '__main__':
    pass