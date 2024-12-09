# AirSketch: Generative Motion to Sketch

This repo contains code for the paper: [AirSketch: Generative Motion to Sketch](https://arxiv.org/abs/2407.08906) from NeurIPS 2024. 

## DATASETS   
Our hand motion datasets can be downloaded from the links below:  
- [Real hand motion dataset](https://drive.google.com/drive/folders/1VPZqrhUB2zJJ4Tp20DgtM2gwp2pBzqz1?usp=sharing)  
- Synthetic hand motion dataset: [1](https://zenodo.org/records/14286297) [2](https://zenodo.org/records/14284049)

Quick, Draw! data can be downloaded with:  
```
python download_qd_all.py
```
This should download ndjsons for samples in the categories listed in `quickdaw_categories.json` into the directory `datasets/quickdraw_ndjsons_raw_all`.

Weights will be added!

## ENVIRONMENT
```
conda create -n airsketch python=3.10
conda activate airsketch

pip install -r requirements.txt
```

## DATA PREPARATION 
### Prepare Quick, Draw! dataset

To prepare the Quick, Draw! dataset after downloading as described in **Datasets**, run: 
```
python data/preprocess.py --quickdraw
```
This should create a filtered, preprocessed dataset in the directory `datasets/preprocessed_filtered_quickdraw_hf_dataset` and a .json file called `datasets/sketches_full.json`.

### Prepare hand-tracked sketches 

To obtain hand-tracked sketches from the hand motion video datasets, first download `hand_landmarker.task` (if not already downloaded), which can be found [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) (more information can be found in [Google AI's 
Hand landmarks detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models)). Then run:
```
python data/preprocess.py \
    --handtracking \
    --video_directory [PATH TO VIDEO DIRECTORY] \
    --dataset_type ["real" or "synthetic"]
```
This should create a directory called `[DATASET_TYPE]_handtracked_sketches` containing the hand-tracked sketches, and a directory called `[DATASET_TYPE]_handtracked_sketch_coordinates` containing the coordinates for these sketches.

## TRAINING:  
To train on the Quick, Draw! dataset with all augmentations applied, prepare the Quick, Draw! dataset as described in **Data Preparation**. Then run:
```
accelerate launch 
    --config_file utils/accelerate_default_config.yaml \
    train_augs2s_controlnet.py \
         --root=datasets/preprocessed_filtered_quickdraw_hf_dataset \
         --run_name=YOUR_RUN_NAME \
         --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
         --sketch_json_path=datasets/sketches_full.json \
         --output_dir=./runs/YOUR_RUN_NAME \
         --mixed_precision=fp16 \
         --validation_steps=2500 \
         --resolution=256 \
         --learning_rate=2e-5 \
         --train_batch_size=8 \
         --dataloader_num_workers=2 \
         --gradient_accumulation_steps=4 \
         --seed=42 \
         --report_to=wandb \
         --enable_xformers_memory_efficient_attention \
         --prompt_format='a black and white sketch of a {}' \
         --line_width=5 \
         --checkpointing_steps=2500 \
         --filter_by_clip_score_percentile=95 \
         --proportion_empty_prompts=0.25 \
         --do_random_erasing
```

To train on the Quick, Draw! dataset with varied sketch line thickness and color, use `train_aug2s_controlnet2.py`.

## INFERENCE AND EVALUATION:  
To run inference and evaluation, first prepare the hand-tracked sketches as described in **Data Preparation**, then run:
```
python inference_controlnet.py \
    --ckpt=PATH_TO_YOUR_CKPT/controlnet \
    --root=PATH_TO_SKETCHES \
    --save_dir=DIRNAME
```
To run inference without evaluation, use flag `--no_eval`.

## CITATION
```
@inproceedings{lim2024airsketch,
      title={AirSketch: Generative Motion to Sketch}, 
      author={Hui Xian Grace Lim and Xuanming Cui and Yogesh S Rawat and Ser-Nam Lim},
      booktitle={NeurIPS}
      year={2024}
}
```
