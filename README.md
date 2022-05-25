# ASSET: Autoregressive Semantic Scene Editing with Transformers at High Resolutions

[Difan Liu](https://people.cs.umass.edu/~dliu/), Sandesh Shetty, [Tobias Hinz](http://www.tobiashinz.com/), [Matthew Fisher](https://techmatt.github.io/), [Richard Zhang](https://richzhang.github.io/), [Taesung Park](https://taesung.me/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/)

UMass Amherst and Adobe Research

ACM Transactions on Graphics (SIGGRAPH 2022)

![teaser](https://people.cs.umass.edu/~dliu/projects/ASSET/resources/teaser.png)

### [Project page](https://people.cs.umass.edu/~dliu/projects/ASSET/) |   [Paper](https://arxiv.org/abs/2205.12231)


## Requirements

- The code has been tested with PyTorch 1.7.1 (GPU version) and Python 3.6.12. 
- Python packages: OpenCV (4.5.1), [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) (1.4.2), omegaconf (2.0.6), [einops](https://github.com/arogozhnikov/einops) (0.3.0), PyYAML (5.4.1), [test-tube](https://github.com/williamFalcon/test-tube) (0.7.5), [albumentations](https://github.com/albumentations-team/albumentations) (1.0.0), [transformers](https://github.com/huggingface/transformers) (4.10.0).

## Testing

### Flickr-Landscape
Download [pretrained model](https://www.dropbox.com/s/5kyov71ko340ra0/landscape.zip?dl=0), replace the config keys in `configs/landscape_test.yaml` with the path of pretrained model, run
```python
python test.py -t configs/landscape_test.yaml -i data_test/landscape_input.jpg -s data_test/landscape_seg.png -m data_test/landscape_mask.png -c waterother -n water_reflection
```

### COCO-Stuff
Download [pretrained model](https://www.dropbox.com/s/np1pljvxck918t8/coco.zip?dl=0), replace the config keys in `configs/coco_test.yaml` with the path of pretrained model, run
```python
python test.py -t configs/coco_test.yaml -i data_test/coco_input.png -s data_test/coco_seg.png -m data_test/coco_mask.png -c pizza -n coco_pizza
```

## Training

### Datasets

- The *Flickr-Landscape* dataset is not sharable due to license issues. But the images were scraped from [Mountains Anywhere](https://flickr.com/groups/62119907@N00/) and [Waterfalls Around the World](https://flickr.com/groups/52241685729@N01/), using the [Python wrapper for the Flickr API](https://github.com/alexis-mignon/python-flickr-api). Please contact [Taesung Park](http://taesung.me/) with title "Flickr Dataset for Swapping Autoencoder" for more details. We include an example from the Flickr-Landscape dataset in `data/landscape` so you can run the training without preparing the dataset.
- For the *COCO-Stuff* dataset, create a symlink `data/coco/imgs` containing the images from the 2017 split in `train2017` and `val2017`, and their annotations in `data/coco/annotations`. Files can be obtained from the [COCO webpage](https://cocodataset.org/). In addition, we use the [Stuff+thing PNG-style annotations on COCO 2017 trainval](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) annotations from [COCO-Stuff](https://github.com/nightrome/cocostuff), which should be placed under `data/coco/cocostuffthings`.

### Flickr-Landscape
Train a VQGAN with:
```python
python main.py --base configs/landscape_vqgan.yaml -t True --gpus -1 --num_gpus 8 --save_dir <path to ckpt>
```

In `configs/landscape_guiding_transformer.yaml`, replace the config key `model.params.first_stage_config.params.ckpt_path` with the pretrained VQGAN, replace the config key `model.params.cond_stage_config.params.ckpt_path` with [landscape_VQ_seg_model.ckpt](https://www.dropbox.com/s/mypovc951nkiv6u/landscape_VQ_seg_model.ckpt?dl=0) (check [Taming Transformers](https://github.com/CompVis/taming-transformers) for the training of VQ_seg_model), train the guiding transformer at 256 resolution with:
```python
python main.py --base configs/landscape_guiding_transformer.yaml -t True --gpus -1 --num_gpus 8 --user_lr 3.24e-5 --save_dir <path to ckpt>
```

In `configs/landscape_SGA_transformer_512.yaml`, replace the config key `model.params.guiding_ckpt_path` and `model.params.ckpt_path` with the pretrained guiding transformer, finetune the SGA transformer at 512 resolution with: 
```python
python main.py --base configs/landscape_SGA_transformer_512.yaml -t True --gpus -1 --num_gpus 8 --user_lr 1.25e-5 --save_dir <path to ckpt>
```

In `configs/landscape_SGA_transformer_1024.yaml`, replace the config key `model.params.guiding_ckpt_path` with the pretrained guiding transformer, replace the config key `model.params.ckpt_path` with the SGA transformer finetuned at 512 resolution, finetune the SGA transformer at 1024 resolution with:
```python
python main.py --base configs/landscape_SGA_transformer_1024.yaml -t True --gpus -1 --num_gpus 8 --user_lr 5e-6 --save_iters 4000 --val_iters 16000 --accumulate_bs 4 --save_dir <path to ckpt>
```

### COCO-Stuff
Train a VQGAN with:
```python
python main.py --base configs/coco_vqgan.yaml -t True --gpus -1 --num_gpus 2 --save_dir <path to ckpt>
```

In `configs/coco_guiding_transformer.yaml`, replace the config key `model.params.first_stage_config.params.ckpt_path` with the pretrained VQGAN, replace the config key `model.params.cond_stage_config.params.ckpt_path` with [coco_VQ_seg_model.ckpt](https://www.dropbox.com/s/us0qncvbh70nq3g/coco_VQ_seg_model.ckpt?dl=0), train the guiding transformer at 256 resolution with:
```python
python main.py --base configs/coco_guiding_transformer.yaml -t True --gpus -1 --num_gpus 8 --user_lr 3.24e-5 --save_dir <path to ckpt>
```

In `configs/coco_SGA_transformer_512.yaml`, replace the config key `model.params.guiding_ckpt_path` and `model.params.ckpt_path` with the pretrained guiding transformer, finetune the SGA transformer at 512 resolution with: 
```python
python main.py --base configs/coco_SGA_transformer_512.yaml -t True --gpus -1 --num_gpus 8 --user_lr 1.25e-5 --save_dir <path to ckpt>
```
## BibTex:
```
@article{liu2022asset,
author = {Liu, Difan and Shetty, Sandesh and Hinz, Tobias and Fisher, Matthew and Zhang, Richard and Park, Taesung and Kalogerakis, Evangelos},
title = {ASSET: Autoregressive Semantic Scene Editing with Transformers at High Resolutions},
year = {2022},
volume = {41},
number = {4},
journal = {ACM Trans. Graph.},}
```
## Acknowledgment
Our code is developed based on [Taming Transformers](https://github.com/CompVis/taming-transformers).

## Contact
To ask questions, please [email](mailto:dliu@cs.umass.edu).
