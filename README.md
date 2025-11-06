# Learning spatial-channel feature refiners via wavelet-based linear mixed basis for low-light image enhancement
Jianming Zhang, Jia Jiang, Zhijian Feng , Jin Wang

[paper](https://authors.elsevier.com/a/1lw1f3OAb9Gq81)

# Quick Run

To test the pretrained models of enhancing on your own images, run

```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```

**All pretrained models can be downloaded at https://pan.baidu.com/s/1xHLRMMKgEMEKAGRveZsqWw?pwd=by7x**  

## Test (Evaluation)  

- To test the PSNR, SSIM and LPIPS of low-light image enhancement results, see [evaluation.py](./evaluation.py) and run

```
python evaluation.py -dirA images_folder_path -dirB images_folder_path -type image_data_type --use_gpu use_gpu_or_not
```

## Train  

To train the restoration models of low-light image enhancement, see [train.py](./train.py) and run 

```
python train.py
```

# Dataset & Pretained Model

1. The LOL-v1 and LOL-v2 datasets analyzed during the current study are available in [LOL](https://daooshee.github.io/BMVC2018website/)

2. The MIT-Adobe FiveK dataset analyzed during the current study is available in [5k](https://data.csail.mit.edu/graphics/fivek/)

3. The DICM, LIME, MEF, NPE and VV datasets analyzed during the current study are available in [non-reference datasets](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T).
    We provie pretrained models:
    ``pretrained_model/models/best_Epoch_LOL.pth``

    ``pretrained_model/models/best_Epoch_LOLv2real.pth``

    ``pretrained_model/models/best_Epoch_LOLv2syn.pth``

    ``pretrained_model/models/best_Epoch_FiveK.pth``

# Cite This Paper
If you find our work or code helpful, or your research benefits from this repo, please cite our paper.

@article{ZHANG2025114567,
title = {Learning spatial-channel feature refiners via wavelet-based linear mixed basis for low-light image enhancement},
journal = {Knowledge-Based Systems},
volume = {330},
pages = {114567},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114567},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125016065},
author = {Jianming Zhang and Jia Jiang and Zhijian Feng and Jin Wang},
}