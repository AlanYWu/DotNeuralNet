# DotNeuralNet

**Light-weight Neural Network for Optical Braille Recognition in the wild & on the book.**

- Classified multi label one-hot encoded labels for raised dots.
- Pseudo-labeled Natural Scene Braille symbols.
- Trained single stage object detection YOLO models for Braille symbols.

### Repository Structure

```
DotNeuralNet
ㄴ assets - example images and train/val logs
ㄴ dataset
  ㄴ AngelinaDataset - book background
  ㄴ braille_natural - natural scene background
  ㄴ DSBI - book background
  ㄴ KaggleDataset - arbitrary 6 dots
  ㄴ yolo.yaml - yolo dataset config
ㄴ src
  ㄴ utils
    ㄴ angelina_utils.py
    ㄴ braille_natural_utils.py
    ㄴ dsbi_utils.py
    ㄴ kaggle_utils.py
  ㄴ crop_bbox.py
  ㄴ dataset.py
  ㄴ model.py
  ㄴ pseudo_label.py
  ㄴ train.py
  ㄴ visualize.py
ㄴ weights
  ㄴ yolov5_braille.pt # yolov5-m checkpoint
  ㄴ yolov8_braille.pt # yolov8-m checkpoint
```

### Result

- Inferenced result of yolov8-m model on validation subset.
  ![yolov8 img](./assets/result_yolov8.png)
- Inferenced result of yolov5-m model on validation subset.
  ![yolov5 img](./assets/result_yolov5.png)

### Logs

- Train / Validation log of yolov8-m model
  ![yolov8 log](./assets/log_yolov8_long.png)
- Train / Validation log of yolov5-m model available at [🔗 WandB](https://wandb.ai/snoop2head/YOLOv5/runs/mqvmh4nc)
  ![yolov8 log](./assets/log_yolov5.png)

### Installation

CV2 and Yolo Dependency Installation

```shell
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### References

```
@article{Li2018DSBIDouble-SidedBraille,
    title   = {DSBI: Double-Sided Braille Image Dataset and Algorithm Evaluation for Braille Dots Detection},
    author  = {Renqiang Li, Hong Liu, Xiangdong Wan, Yueliang Qian},
    journal = {ArXiv},
    year    = {2018},
    volume  = {abs/1811.1089}
}
```

```
@article{Ovodov2021OpticalBrailleRecog,
    title   = {Optical Braille Recognition Using Object Detection CNN},
    author  = {Ilya G. Ovodov},
    journal = {2021 IEEE/CVF International Conference on Computer Vision Workshops},
    year    = {2021},
    pages   = {1741-1748}
}
```

```
@article{lu2022AnchorFreeBrailleCharac
    title   = {Anchor-Free Braille Character Detection Based on Edge Feature in Natural Scene Images},
    author  = {Liqiong Lu, Dong Wu, Jianfang Xiong, Zhou Liang and Faliang Huang},
    journal = {Computational Intelligence and Neuroscience},
    year    = {2022},
    url     = {https://www.hindawi.com/journals/cin/2022/7201775}
}
```
