import datasets
from datasets import load_from_disk

import os
import re
import json
import random
import numpy as np
from rdp import rdp
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
from .helpers import (generate_sketch_image, 
                      normalize_sketch,
                      increase_line_width,
                      transform_sketch
                      )
from torchvision.transforms.functional import hflip

class AugmentedSketch2SketchAdapterDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.filenames = []
        self.categories = []
        self.indices = []
        self.toTensor = transforms.ToTensor()
        self.category_stats = {}
        # if type(args.root) == list:
        #     dataset_list = []
        #     for root in args.root:
        #         dataset_list.append(load_from_disk(root)['train']) 
        #         self.category_stats.update(json.load(open(os.path.join(root, 'category_stats.json'), 'r'))) 
        #     self.dataset = datasets.concatenate_datasets(dataset_list)
        # else:
        self.dataset = load_from_disk(args.root)
        self.category_stats = json.load(open(os.path.join(self.args.root, 'category_stats.json'), 'r'))
        self.resize = transforms.Resize((args.resolution, args.resolution))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        datapoint = self.dataset[idx]
        sketch, category = datapoint['drawing'], datapoint['category']
        sketch = [np.array([(x,y) for x,y,t in stroke]) for stroke in sketch] # don't want t 
        gt = generate_sketch_image(sketch, (self.args.resolution, self.args.resolution), line_width=self.args.line_width)

        # background: 1, sketch: 0
        gt = self.toTensor(gt) 

        if self.args.no_augmentation:
            return {"pixel_values": gt, 
                "conditioning_pixel_values": 1-gt, 
                "prompt": self.args.prompt_format.format(category), 
                "coordinate_sequence": None,
                **self.embeds_dict[category]}

        jitter_range = rescale_range = move_range = None
        erase_part_range = None
        do_random_oscillation = False
        per_point_spike_prob = None
        rescale_range = move_range = None
        add_random_false_strokes = add_transition_line = add_start_end_false_strokes = False

        structural_aug = np.random.choice(self.args.structural_augmentations)
        local_aug = np.random.choice(self.args.local_augmentations) # 'jitter', 'oscillation', 'spikes'

        if structural_aug == 'shrink_and_relocate':
            sketch = shrink_and_relocate(sketch, 0.6)
        elif structural_aug == 'rescale':
            rescale_range = (0.7, 0.9)
        elif structural_aug == 'move':
            move_range = (0.1, 0.2)

        if local_aug == 'jitter':
            jitter_range = (0, 0.01) 
        elif local_aug == 'oscillation':
            do_random_oscillation = True
        elif local_aug == 'spikes':
            per_point_spike_prob = 0.1

        if self.args.do_random_erasing and np.random.rand() > 0.5 :
            erase_part_range=(0.2, 0.6)
        
        if "random" in self.args.false_strokes:
            add_random_false_strokes = np.random.rand() > 0.25
        if "transition" in self.args.false_strokes:
            add_transition_line = np.random.rand() > 0.25
        if "start_end" in self.args.false_strokes:
            add_start_end_false_strokes = True
            
        augmented_sketch = augment_strokes(sketch,
                                            rescale_range=rescale_range, 
                                            jitter_range=jitter_range, 
                                            move_range=move_range, 
                                            add_transition_line=add_transition_line, 
                                            erase_part_range=erase_part_range, 
                                            do_random_oscillation=do_random_oscillation,
                                            per_point_spike_prob=per_point_spike_prob,
                                            add_random_false_strokes=add_random_false_strokes,
                                            add_start_end_false_strokes=add_start_end_false_strokes)
        aug = generate_sketch_image(augmented_sketch, (256, 256), line_width=self.args.line_width)

        if self.args.use_coordinate_sequence_conditioning:
            augmented_sketch = [rdp(stroke, 0.02) for stroke in augmented_sketch]
            augmented_sketch = [[(x,y,t) for stroke in augmented_sketch for x,y,t in stroke]]
        else:
            augmented_sketch = None

        aug = self.toTensor(aug)
        aug = 1 - aug # by default the conditioning should be 0 background

        return {"pixel_values": gt, 
                "conditioning_pixel_values": aug, 
                "prompt": self.args.prompt_format.format(category), 
                "coordinate_sequence": augmented_sketch,
                **self.embeds_dict[category]}

    def prepare_train_dataset(self, text_encoders, tokenizers, accelerator):
        
        def filter_func(example):
            
            if int(example['id']) < 100:
                return False

            if self.args.held_out_categories and example['category'] in self.args.held_out_categories:
                return False

            if self.args.validation_samples is not None:
                for cat, idx in self.args.validation_samples:
                    if example["category"] == cat and int(example["id"]) == int(idx):
                        return False

            if self.args.filter_by_clip_score_percentile is not None:
                threshold = self.category_stats[example['category']]['percentile'][str(self.args.filter_by_clip_score_percentile)]
                if example['clip_score'] < threshold:
                    return False

            return True

        with accelerator.main_process_first():

            self.dataset = self.dataset.filter(filter_func, num_proc=os.cpu_count())
            self.dataset = self.dataset.shuffle(seed=self.args.seed)
            if self.args.max_train_samples is not None:
                self.dataset = self.dataset.select(range(self.args.max_train_samples))

            self.compute_embeds(text_encoders, tokenizers, accelerator)

    def compute_embeds(self, text_encoders, tokenizers, accelerator, is_train=True):
        unique_categories = list(self.category_stats.keys())
        unique_categories = sorted(unique_categories)
        prompts = [self.args.prompt_format.format(category) for category in unique_categories]
        original_size = (self.args.resolution, self.args.resolution)
        target_size = (self.args.resolution, self.args.resolution)
        crops_coords_top_left = (self.args.crops_coords_top_left_h, self.args.crops_coords_top_left_w)

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompts, text_encoders, tokenizers, self.args.proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to('cpu')
        add_text_embeds = add_text_embeds.to('cpu')
        add_time_ids = add_time_ids.repeat(len(prompts), 1)
        add_time_ids = add_time_ids.to('cpu', dtype=prompt_embeds.dtype)
        self.embeds_dict = {}
        for i, category in enumerate(unique_categories):
            self.embeds_dict[category] = {"prompt_embeds": prompt_embeds[i].clone(), "text_embeds": add_text_embeds[i].clone(), "time_ids": add_time_ids[i]}

class AugmentedSketch2SketchAdapterDataset2(AugmentedSketch2SketchAdapterDataset):
    def __init__(self, args):
        super().__init__(args)
        self.color_rgb_palette = json.load(open('color_palette.json', 'r'))
        self.line_thickness_map = {"thin": 2, "medium thick": 5, "thick": 10}

        original_size = (self.args.resolution, self.args.resolution)
        target_size = (self.args.resolution, self.args.resolution)
        crops_coords_top_left = (self.args.crops_coords_top_left_h, self.args.crops_coords_top_left_w)

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor(add_time_ids)

        self.add_time_ids = add_time_ids.to('cpu')

    def __getitem__(self, idx):
        
        datapoint = self.dataset[idx]
        sketch, category = datapoint['drawing'], datapoint['category']
        sketch = [np.array([(x,y) for x,y,t in stroke]) for stroke in sketch] # don't want t 
        color = random.choice(list(self.color_rgb_palette.keys()))
        line_thickness = random.choice(list(self.line_thickness_map.keys()))

        prompt = f"a simple sketch of a {category}, line drawing, white background, {color} lines, {line_thickness} lines"
        gt = generate_sketch_image(sketch, (self.args.resolution, self.args.resolution), line_width=self.line_thickness_map[line_thickness], line_color=tuple(self.color_rgb_palette[color]))

        # background: 1, sketch: 0
        gt = self.toTensor(gt) 

        if self.args.no_augmentation:
            return {"pixel_values": gt, 
                "conditioning_pixel_values": 1-gt, 
                "prompt": prompt, 
                "time_ids": self.add_time_ids,
                "coordinate_sequence": None}

        jitter_range = rescale_range = move_range = None
        erase_part_range = None
        do_random_oscillation = False
        per_point_spike_prob = None
        rescale_range = move_range = None
        add_random_false_strokes = add_transition_line = add_start_end_false_strokes = False

        structural_aug = np.random.choice(self.args.structural_augmentations)
        local_aug = np.random.choice(self.args.local_augmentations) # 'jitter', 'oscillation', 'spikes'

        # TODO: investigate the negatively impacting augmentations
        if structural_aug == 'shrink_and_relocate':
            sketch = shrink_and_relocate(sketch, 0.6)
        elif structural_aug == 'rescale':
            rescale_range = (0.7, 0.9)
        elif structural_aug == 'move':
            move_range = (0.1, 0.2)

        if local_aug == 'jitter':
            jitter_range = (0, 0.01) 
        elif local_aug == 'oscillation':
            do_random_oscillation = True
        elif local_aug == 'spikes':
            per_point_spike_prob = 0.1

        if self.args.do_random_erasing and np.random.rand() > 0.5 :
            erase_part_range=(0.2, 0.6)
        
        if "random" in self.args.false_strokes:
            add_random_false_strokes = np.random.rand() > 0.25
        if "transition" in self.args.false_strokes:
            add_transition_line = np.random.rand() > 0.25
        if "start_end" in self.args.false_strokes:
            add_start_end_false_strokes = True
            
        augmented_sketch = augment_strokes(sketch,
                                            rescale_range=rescale_range, 
                                            jitter_range=jitter_range, 
                                            move_range=move_range, 
                                            add_transition_line=add_transition_line, 
                                            erase_part_range=erase_part_range, 
                                            do_random_oscillation=do_random_oscillation,
                                            per_point_spike_prob=per_point_spike_prob,
                                            add_random_false_strokes=add_random_false_strokes,
                                            add_start_end_false_strokes=add_start_end_false_strokes)
        aug = generate_sketch_image(augmented_sketch, (256, 256), line_width=self.args.line_width)

        if self.args.use_coordinate_sequence_conditioning:
            augmented_sketch = [rdp(stroke, 0.02) for stroke in augmented_sketch]
            augmented_sketch = [[(x,y,t) for stroke in augmented_sketch for x,y,t in stroke]]
        else:
            augmented_sketch = None

        aug = self.toTensor(aug)
        aug = 1 - aug # by default the conditioning should be 0 background

        return {"pixel_values": gt, 
                "conditioning_pixel_values": aug, 
                "prompt": prompt, 
                "time_ids": self.add_time_ids,
                "coordinate_sequence": augmented_sketch}

    def prepare_train_dataset(self, accelerator):
        
        def filter_func(example):
            
            if int(example['id']) < 100:
                return False

            if self.args.held_out_categories and example['category'] in self.args.held_out_categories:
                return False

            if self.args.validation_samples is not None:
                for cat, idx in self.args.validation_samples:
                    if example["category"] == cat and int(example["id"]) == int(idx):
                        return False

            if self.args.filter_by_clip_score_percentile is not None:
                threshold = self.category_stats[example['category']]['percentile'][str(self.args.filter_by_clip_score_percentile)]
                if example['clip_score'] < threshold:
                    return False

            return True

        with accelerator.main_process_first():

            self.dataset = self.dataset.filter(filter_func, num_proc=os.cpu_count())
            self.dataset = self.dataset.shuffle(seed=self.args.seed)
            if self.args.max_train_samples is not None:
                self.dataset = self.dataset.select(range(self.args.max_train_samples))

    
class AugmentedS2SControlNetSD15Dataset(AugmentedSketch2SketchAdapterDataset):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def compute_embeds(self, text_encoder, tokenizer, accelerator, is_train=True):
        unique_categories = list(self.category_stats.keys())
        unique_categories = sorted(unique_categories)
        prompts = [self.args.prompt_format.format(category) for category in unique_categories]
        prompts += ""
        input_ids = tokenizer(
            prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(accelerator.device)
        text_embeds = text_encoder(input_ids, return_dict=False)[0].cpu()
        self.embeds_dict = {}
        for i, category in enumerate(unique_categories):
            self.embeds_dict[category] = {"encoder_hidden_states": text_embeds[i].clone()}
        self.embeds_dict[""] = {"encoder_hidden_states": text_embeds[-1].clone()}

def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def collate_fn(examples, args):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example['conditioning_pixel_values'] for example in examples], dim=0)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([example["prompt_embeds"] for example in examples])

    add_text_embeds = torch.stack([example["text_embeds"] for example in examples])
    add_time_ids = torch.stack([example["time_ids"] for example in examples])

    coordinate_sequence = [example["coordinate_sequence"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        "coordinate_sequence": coordinate_sequence
    }

def collate_fn2(examples, args):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example['conditioning_pixel_values'] for example in examples], dim=0)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    add_time_ids = torch.stack([example["time_ids"] for example in examples])

    coordinate_sequence = [example["coordinate_sequence"] for example in examples]
    prompts = [example["prompt"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt": prompts,
        "unet_added_conditions": {"time_ids": add_time_ids},
        "coordinate_sequence": coordinate_sequence
    }

def collate_fn_sd15(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example['conditioning_pixel_values'] for example in examples], dim=0)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    encoder_hidden_states = torch.stack([example["encoder_hidden_states"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "encoder_hidden_states": encoder_hidden_states,
    }


def preprocess_video_sketch_for_validation(category, sketch_id, sketch_dict, args):

    sketch = sketch_dict[category][str(sketch_id)]

    sketch = [np.array([(points[0],points[1]) for points in stroke]) for stroke in sketch] # don't want t 
    gt = generate_sketch_image(sketch, (args.resolution, args.resolution), line_width=args.line_width)
    prompt = args.prompt_format.format(category)

    if args.no_augmentation:
        aug = transforms.ToTensor()(gt)
        aug = 1 - aug
        return aug, gt, prompt, None

    jitter_range = rescale_range = move_range = None
    do_random_oscillation = False
    per_point_spike_prob = None
    rescale_range = move_range = None
    erase_part_range = None
    add_random_false_strokes = add_transition_line = add_start_end_false_strokes = False

    structural_aug = np.random.choice(args.structural_augmentations)
    local_aug = np.random.choice(args.local_augmentations) # 'jitter', 'oscillation', 'spikes'

    if structural_aug == 'shrink_and_relocate':
        sketch = shrink_and_relocate(sketch, 0.6)
    elif structural_aug == 'rescale':
        rescale_range = (0.7, 0.9)
    elif structural_aug == 'move':
        move_range = (0.1, 0.2)

    if local_aug == 'jitter':
        jitter_range = (0, 0.01) 
    elif local_aug == 'oscillation':
        do_random_oscillation = True
    elif local_aug == 'spikes':
        per_point_spike_prob = 0.1
        
    if args.do_random_erasing and np.random.rand() > 0.5:
        erase_part_range=(0.2, 0.6)

    if "random" in args.false_strokes:
        add_random_false_strokes = np.random.rand() > 0.25
    if "transition" in args.false_strokes:
        add_transition_line = np.random.rand() > 0.25
    if "start_end" in args.false_strokes:
        add_start_end_false_strokes = True
        
    augmented_sketch = augment_strokes(sketch,
                                        rescale_range=rescale_range, 
                                        jitter_range=jitter_range, 
                                        move_range=move_range, 
                                        add_transition_line=add_transition_line, 
                                        erase_part_range=erase_part_range, 
                                        do_random_oscillation=do_random_oscillation,
                                        per_point_spike_prob=per_point_spike_prob,
                                        add_random_false_strokes=add_random_false_strokes,
                                        add_start_end_false_strokes=add_start_end_false_strokes)
                                        
    aug = generate_sketch_image(augmented_sketch, (args.resolution, args.resolution), line_width=args.line_width)
    aug = transforms.ToTensor()(aug)
    aug = 1 - aug
    
    return aug, gt, prompt, augmented_sketch


def preprocess_video_sketch_for_validation2(category, sketch_id, sketch_dict, args, color_rgb_palette=None, line_thickness_map=None):

    if color_rgb_palette is None:
        color_rgb_palette = json.load(open('color_palette.json', 'r'))
    if line_thickness_map is None:
        line_thickness_map = {"thin": 2, "medium thick": 5, "thick": 10}

    sketch = sketch_dict[category][str(sketch_id)]

    sketch = [np.array([(points[0],points[1]) for points in stroke]) for stroke in sketch] # don't want t 

    color = random.choice(list(color_rgb_palette.keys()))
    line_thickness = random.choice(list(line_thickness_map.keys()))
    prompt = f"a simple sketch of a {category}, line drawing, white background, {color} lines, {line_thickness} lines"
    gt = generate_sketch_image(sketch, (args.resolution, args.resolution), line_width=line_thickness_map[line_thickness], line_color=tuple(color_rgb_palette[color]))

    if args.no_augmentation:
        aug = transforms.ToTensor()(gt)
        aug = 1 - aug
        return aug, gt, prompt, None

    jitter_range = rescale_range = move_range = None
    do_random_oscillation = False
    per_point_spike_prob = None
    rescale_range = move_range = None
    erase_part_range = None
    add_random_false_strokes = add_transition_line = add_start_end_false_strokes = False

    structural_aug = np.random.choice(args.structural_augmentations)
    local_aug = np.random.choice(args.local_augmentations) # 'jitter', 'oscillation', 'spikes'

    if structural_aug == 'shrink_and_relocate':
        sketch = shrink_and_relocate(sketch, 0.6)
    elif structural_aug == 'rescale':
        rescale_range = (0.7, 0.9)
    elif structural_aug == 'move':
        move_range = (0.1, 0.2)

    if local_aug == 'jitter':
        jitter_range = (0, 0.01) 
    elif local_aug == 'oscillation':
        do_random_oscillation = True
    elif local_aug == 'spikes':
        per_point_spike_prob = 0.1
        
    if args.do_random_erasing and np.random.rand() > 0.5:
        erase_part_range=(0.2, 0.6)

    if "random" in args.false_strokes:
        add_random_false_strokes = np.random.rand() > 0.25
    if "transition" in args.false_strokes:
        add_transition_line = np.random.rand() > 0.25
    if "start_end" in args.false_strokes:
        add_start_end_false_strokes = True
        
    augmented_sketch = augment_strokes(sketch,
                                        rescale_range=rescale_range, 
                                        jitter_range=jitter_range, 
                                        move_range=move_range, 
                                        add_transition_line=add_transition_line, 
                                        erase_part_range=erase_part_range, 
                                        do_random_oscillation=do_random_oscillation,
                                        per_point_spike_prob=per_point_spike_prob,
                                        add_random_false_strokes=add_random_false_strokes,
                                        add_start_end_false_strokes=add_start_end_false_strokes)
                                        
    aug = generate_sketch_image(augmented_sketch, (args.resolution, args.resolution), line_width=args.line_width)
    aug = transforms.ToTensor()(aug)
    aug = 1 - aug
    
    return aug, gt, prompt, augmented_sketch
    
def shrink_and_relocate(sketch, min_scale):
    min_x = min_y = 1.0
    max_x = max_y = 0.0
    scale_factor_x = random.uniform(min_scale, 1.0)
    scale_factor_y = random.uniform(min_scale, 1.0)
    
    for stroke in sketch:
        stroke[:, 0] *= scale_factor_x
        stroke[:, 1] *= scale_factor_y
    
    scaled_min_x = scaled_min_y = 1.0
    scaled_max_x = scaled_max_y = 0.0
    for stroke in sketch:
        for x, y in stroke:
            scaled_min_x = min(scaled_min_x, x)
            scaled_min_y = min(scaled_min_y, y)
            scaled_max_x = max(scaled_max_x, x)
            scaled_max_y = max(scaled_max_y, y)
            
    max_trans_x = 1.0 - scaled_max_x
    max_trans_y = 1.0 - scaled_max_y
    
    trans_x = random.uniform(-scaled_min_x, max_trans_x)
    trans_y = random.uniform(-scaled_min_y, max_trans_y)
    relocated_sketch = [[[x + trans_x, y + trans_y] for x, y in stroke] for stroke in sketch]
    
    return relocated_sketch

def augment_strokes(sketch_data, 
                    rescale_range=None, 
                    jitter_range=None, 
                    move_range=None, 
                    jitter_type='gaussian', 
                    add_transition_line=True, 
                    erase_part_range=None,
                    add_start_end_false_strokes=True,
                    per_point_spike_prob=None,
                    do_random_oscillation=False,
                    add_random_false_strokes=True):

    if erase_part_range:
        if random.random() > 0.3:
            sketch_data = random_erase_whole_parts(sketch_data, portion_range=erase_part_range, part='end')
        else:
            sketch_data = random_erase_whole_parts(sketch_data, portion_range=erase_part_range, part='start')


    augmented_sketch = []

    if add_random_false_strokes and len(sketch_data) > 1:
        n_random_false_strokes = np.random.randint(len(sketch_data)//2)
        for _ in range(n_random_false_strokes):
            start_from_stroke_idx, end_on_stroke_idx = np.random.randint(len(sketch_data)), np.random.randint(len(sketch_data))
            start_from, end_on = sketch_data[start_from_stroke_idx], sketch_data[end_on_stroke_idx]
            start_from, end_on = start_from[np.random.randint(len(start_from))], end_on[np.random.randint(len(end_on))]
            stroke = generate_random_points_between(start_from, end_on)
            sketch_data.insert(np.random.randint(len(sketch_data)), stroke)

    for i, stroke in enumerate(sketch_data):
        stroke = np.array(stroke)
        if rescale_range is not None:

            min_x = min_y = 1.0
            max_x = max_y = 0.0

            for x, y in stroke:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            center_x = sum(x for x, _ in stroke) / len(stroke)
            center_y = sum(y for _, y in stroke) / len(stroke)

            if np.random.randn() > 0.5:
                rescale = (1 + rescale_range[0], 1 + rescale_range[1])
            else:
                rescale = (1 - rescale_range[1], 1 - rescale_range[0])

            scale_factor = random.uniform(*rescale)


            adjusted_scale_factors = []
            for x, y in stroke:
                if x < center_x:
                    adjusted_scale_factors.append((min_x - center_x) / (x - center_x))
                elif x > center_x:
                    adjusted_scale_factors.append((max_x - center_x) / (x - center_x))
                if y < center_y:
                    adjusted_scale_factors.append((min_y - center_y) / (y - center_y))
                elif y > center_y:
                    adjusted_scale_factors.append((max_y - center_y) / (y - center_y))
            max_safe_scale = min(adjusted_scale_factors) if adjusted_scale_factors else scale_factor
            scale_factor = min(scale_factor, max_safe_scale)

            stroke = np.array([((x - center_x) * scale_factor + center_x, (y - center_y) * scale_factor + center_y) for x, y in stroke])

        if move_range is not None:
            if np.random.rand() > 0.5:
                move_range_x = (-move_range[1], -move_range[0])
            else:
                move_range_x = move_range
            if np.random.rand() > 0.5:
                move_range_y = (-move_range[1], -move_range[0])
            else:
                move_range_y = move_range

            assert move_range_x[0] <= move_range_x[1], f"move_range_x lower bound {move_range_x[0]} is greater than upper bound {move_range_x[1]}"
            assert move_range_y[0] <= move_range_y[1], f"move_range_y lower bound {move_range_y[0]} is greater than upper bound {move_range_y[1]}"
            min_x = min(stroke, key=lambda p: p[0])[0]
            max_x = max(stroke, key=lambda p: p[0])[0]
            min_y = min(stroke, key=lambda p: p[1])[1]
            max_y = max(stroke, key=lambda p: p[1])[1]
            if move_range_x[0] < 0:
                move_x_range_l = max(move_range_x[0], -min_x)
            else:
                move_x_range_l = min(move_range_x[0], 1 - max_x)

            if move_range_x[1] < 0:
                move_x_range_h = max(move_range_x[1], -min_x)
            else:
                move_x_range_h = min(move_range_x[1], 1 - max_x)       

            if move_range_y[0] < 0:
                move_y_range_l = max(move_range_y[0], -min_y)
            else:
                move_y_range_l = min(move_range_y[0], 1 - max_y)

            if move_range_y[1] < 0:
                move_y_range_h = max(move_range_y[1], -min_y)
            else:
                move_y_range_h = min(move_range_y[1], 1 - max_y)          

            move_x = random.uniform(move_x_range_l, move_x_range_h) 
            move_y = random.uniform(move_y_range_l, move_y_range_h)

            stroke[:, 0] += move_x
            stroke[:, 1] += move_y

        if add_transition_line and i < len(sketch_data) - 1:
            start, end = stroke[-1], sketch_data[i+1][0]
            transition_points = generate_random_points_between(start, end)
            stroke = np.concatenate([stroke, transition_points]) 

        if add_start_end_false_strokes:
            if i == 0:
                start = random_point_on_border()
                transition_points_start = generate_random_points_between(start, stroke[0])
                stroke = np.concatenate([transition_points_start, stroke])
            elif i == len(sketch_data) - 1:
                end = random_point_on_border()
                transition_points_end = generate_random_points_between(stroke[-1], end)
                stroke = np.concatenate([stroke, transition_points_end])

        if jitter_range is not None:

            if jitter_type == 'uniform':
                jitters = np.random.uniform(*jitter_range, (2, len(stroke)))
            elif jitter_type == 'gaussian':
                jitters = np.random.normal(*jitter_range, (2, len(stroke)))
            elif jitter_type == 'student_t':
                jitters = np.random.standard_t(*jitter_range, (2, len(stroke)))

            stroke[:, 0] += jitters[0]
            stroke[:, 1] += jitters[1]

        if per_point_spike_prob is not None and len(stroke) > 3:
            stroke = add_random_spikes(stroke, per_point_spike_prob=per_point_spike_prob)

        if do_random_oscillation and len(stroke) > 3:
            stroke += random_fourier_series(stroke)

        stroke = np.clip(stroke, 0, 1)

        augmented_sketch.append(stroke)

    # if erase_part_range:
    #     if random.random() > 0.3:
    #         augmented_sketch = random_erase_whole_parts(augmented_sketch, portion_range=erase_part_range, part='end')
    #     else:
    #         augmented_sketch = random_erase_whole_parts(augmented_sketch, portion_range=erase_part_range, part='start')

    return augmented_sketch


import random
import math

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def random_point_on_border():
    random_side = random.randint(1,3)
    if random_side == 1:
        return (random.uniform(0,1), 1)
    elif random_side == 2:
        return (1, random.uniform(0.5,1))
    elif random_side == 3:
        return (0, random.uniform(0.5,1))
    else:
        raise ValueError("Invalid random side")


def random_erase_whole_parts(sketch, portion_range=(0.2, 0.6), part='end'):
    portion = random.uniform(*portion_range)
    total_num_points = sum(len(stroke) for stroke in sketch)
    modified_sketch = []
    if part == 'end':
        num_points_to_keep = int((1-portion) * total_num_points)
        for stroke in sketch:
            if num_points_to_keep > 0:
                if len(stroke) > num_points_to_keep:
                    modified_sketch.append(stroke[:num_points_to_keep])
                    break
                else:
                    modified_sketch.append(stroke)
                    num_points_to_keep -= len(stroke)
            else:
                break
    elif part == 'start':
        num_points_to_erase = int(portion * total_num_points)
        for stroke in sketch:
            if num_points_to_erase > 0:
                if len(stroke) > num_points_to_erase:
                    modified_sketch.append(stroke[num_points_to_erase:])
                    num_points_to_erase = 0
                else:
                    num_points_to_erase -= len(stroke)
            else:
                modified_sketch.append(stroke)
    
    return modified_sketch

def generate_random_points_between(start, end, sample_dist='gaussian', sample_dist_stats=(0.02, 0.02)):

    total_distance = calculate_distance(start, end)
    direction = (end[0] - start[0], end[1] - start[1])
    
    points = [start]
    current_point = start
    cumulative_distance = 0
    
    while cumulative_distance < total_distance:
        interval_distance = abs(random.gauss(*sample_dist_stats))
        
        cumulative_distance += interval_distance
        
        if cumulative_distance < total_distance:

            new_point = (current_point[0] + (interval_distance / total_distance) * direction[0],
                         current_point[1] + (interval_distance / total_distance) * direction[1])
            
            points.append(new_point)
            current_point = new_point
        else:
            break
            
    points.append(end)
    
    return np.array(points)

def compute_normals(points):

    normals = []
    for i in range(1, len(points)-1):
        t1 = np.array(points[i]) - np.array(points[i-1])
        t2 = np.array(points[i+1]) - np.array(points[i])
        tangent = (t1 + t2) / 2
        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
        normal = np.array([-tangent[1], tangent[0]])
        normals.append(normal)
    normals = [normals[0]] + normals + [normals[-1]]
    return normals
    
def random_fourier_series(points, amplitude_range=(0.05,0.15), frequency_range=(2,5), num_terms=20):

    normals = compute_normals(points)
    euc_length = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))[-1]
    x = np.linspace(0, 1, len(points))
    oscillation = np.zeros_like(x)
    for _ in range(num_terms):
        amplitude = np.random.uniform(*amplitude_range) / num_terms
        frequency = np.random.uniform(*frequency_range) * euc_length  # Base frequency multipliers
        phase = np.random.uniform(0, 2 * np.pi)
        # Random choice between sine and cosine for each term
        f = amplitude * np.sin(2 * np.pi * frequency * x + phase)
        if np.random.rand() > 0.5:
            oscillation += amplitude * np.sin(2 * np.pi * frequency * x + phase)
        else:
            oscillation += amplitude * np.cos(2 * np.pi * frequency * x + phase)
    return oscillation[:, None].repeat(2, axis=1) * normals

def cubic_bezier(p0, p1, p2, p3, t):
    return (1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + \
           3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)

def calculate_distances(points):
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Include the starting point
    return distances

def add_random_spikes(points, per_point_spike_prob=0.01, spike_length_range=(0.05, 0.2), spike_amplitude_range=(0, 0.7), dist='gaussian'):
    points = np.array(points)
    distances = calculate_distances(points)
    if distances[-1] < spike_length_range[0]:
        return points
    new_points = []
    i = 0

    mode = np.random.choice(['bezier', 'spike'])

    while i < len(points) - 1:
        if np.random.rand() > per_point_spike_prob:
            new_points.append(points[i])
            i += 1
            continue
            
        remaining_length = distances[-1] - distances[i]
        curve_length = np.random.uniform(*spike_length_range)
        if remaining_length < curve_length:
            new_points.extend(points[i:])
            break

        # Determine length of the next curve
        end_distance = distances[i] + curve_length
        end_index = np.searchsorted(distances, end_distance, side='right') - 1
        end_index = min(end_index, len(points) - 1)  # Ensure within bounds

        # Generate control points and the curve
        p0 = points[i]
        p3 = points[end_index]
        vector = p3 - p0
        norm_vector = np.linalg.norm(vector)
        if norm_vector == 0:
            new_points.append(p0)
            i += 1
            continue

        if mode == 'bezier':

            # Control points with some randomness
            control_factor = np.random.uniform(0,1)
            p1 = p0 + control_factor * vector + np.random.normal(*spike_amplitude_range) * np.array([-vector[1], vector[0]])
            p2 = p0 + (1-control_factor) * vector + np.random.normal(*spike_amplitude_range) * np.array([-vector[1], vector[0]])
            
            # Compute the Bézier curve
            t_values = np.linspace(0, 1, max(20, int(norm_vector * 10)))
            bezier_points = [cubic_bezier(p0, p1, p2, p3, t) for t in t_values]

            new_points.extend(bezier_points[:-1])  # Exclude last point to avoid duplication
        else:
            m = (p3[1] - p0[1]) / (p3[0] - p0[0] + 1e-6)
            b = p0[1] - m * p0[0]
            x_control = np.random.uniform(p0[0], p3[0])
            if dist == 'gaussian':
                y_drift = np.random.normal(*spike_amplitude_range)
            elif dist == 'uniform':
                y_drift = np.random.uniform(*spike_amplitude_range)
            y_control = m * x_control + b + y_drift * norm_vector
            new_points.append((x_control, y_control))

        i = end_index

    return np.array(new_points)


def preprocess_sketch_for_inference(filename, sketch_dict, args):

    transformed_strokes = transform_sketch(sketch_data)
    xyt_strokes = []
    for stroke in transformed_strokes:
        xyt_pairs = [(x,y,t) for x,y,t in zip(*stroke)]

    resize = transforms.Resize((args.resolution, args.resolution))
    pattern = r'^([^0-9]+)(\d+)\.png'
    match = re.match(pattern, filename.replace(' ', ''))
    category = match.group(1)
    idx = match.group(2)
    prompt = args.prompt_format.format(category)
    inp = Image.open(os.path.join(args.root, filename))
    inp = transforms.ToTensor()(inp)
    inp = 1 - inp
    inp = inp[:,52:52+1028, 37:37+1028].clone()
    inp = normalize_sketch(inp)
    inp = resize(inp)
    if 'real' not in args.root:
        inp = hflip(inp)

    sketch = sketch_dict[category][idx]
    gt = generate_sketch_image(sketch, (args.resolution, args.resolution), line_width=2)
    if args.line_width > 2:
        gt = transforms.ToPILImage()(1-increase_line_width(1-transforms.ToTensor()(gt).unsqueeze(0), kernel_size=args.line_width - 2).squeeze(0))
        inp = increase_line_width(inp.unsqueeze(0), kernel_size=args.line_width - 2).squeeze(0)

    return inp, gt, prompt