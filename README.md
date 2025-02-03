# WS-WS: Weakly Supervised Waste Sorting Segmentation Benchmark

## Overview
This repository provides scripts for training and evaluating weakly supervised segmentation models for the Seruso Dataset. It includes implementations for:
- **Standard Classifier (GradCAM, GradCAM++, LayerCAM)**
- **PuzzleCAM**
- **POF-CAM**

**Note:** The instructions in this README apply to all methods **except** for *WeakTr*, which is implemented on a different branch.

## Dataset Structure
The dataset is structured into folders as follows:

```
train_val_dataset/
├── training/
│   ├── before/
│   │   ├── video_000/
│   │   │   ├── frame_0001
│   │   │   ├── frame_0002
│   │   │   ├── ...
│   │   ├── video_001/
│   │   │   ├── frame_0001
│   │   │   ├── frame_0002
│   │   │   ├── ...
│   ├── after/
│   ├── bg/ (optional)
├── validation/
│   ├── before/
│   ├── after/
│   ├── bg/ (optional)

test_set/
├── images/
│   ├── before/
│   ├── after/
├── masks/
│   ├── before/
│   ├── after/
```

For the **POF-CAM** method, optical flow data should follow the same structure as the dataset.

## Experiment Results & Logs
All models, logs, and results are saved in the `experiments/` folder, which includes:
- Trained models
- Coarse masks
- Evaluation results

## Training the Classifier
To train a classifier, modify the YAML configuration file at:
```
configs/run_all_experiments_train.yaml
```
with your preferred settings. Then, run the following command:

```
python run_all_experiments.py --config configs/run_all_experiments_train.yaml --method standard --cuda_devices 0,1,2
```

Available methods:
- `standard`
- `PuzzleCAM`
- `POF_CAM`

## Evaluating on the Test Set
Modify the evaluation YAML file at:
```
configs/run_all_experiments_evaluate_WS_methods.yaml
```
Select the test set folder, then execute:

```
python run_all_experiments_evaluate_inference.py --config configs/run_all_experiments_evaluate_WS_methods.yaml --method standard --cuda_devices 0,1,2
```

## Generating New Training Masks
To generate new training masks, modify:
```
configs/run_all_experiments_generate_masks.yaml
```
Select the training and validation dataset folders, then run:

```
python run_all_experiments_inference.py --config configs/run_all_experiments_generate_masks.yaml --method standard --cuda_devices 0,1,2
```

This process will save the generated masks in the `experiments/` folder while maintaining the same structure as the dataset.

