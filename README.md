# AdaIN-pytorch
[Korean Instruction](https://grow-up-by-coding.tistory.com/entry/AdaIN-StyleTransfer-%EC%A0%95%EB%A6%AC-%EB%B0%8F-%EC%BD%94%EB%93%9C)

## Dataset
- I used [MSCOCO](https://cocodataset.org/#download) val2014 for content images, and test.zip in [Wikiart images](https://www.kaggle.com/c/painter-by-numbers) for style images.
- 1000 content images and 1000 style images were used. Check `train_content_paths.txt`, `train_style_paths.txt`, `test_content_paths.txt`, `test_style_paths.txt`.

## Install
```
source scripts/install.sh
```

## Demo
```
source scripts/demo.sh
```
- Please download the checkpoint file from this link. [Download](https://drive.google.com/file/d/1TNy__tq0OMGkEOairsgY9S9dVoThmIjn/view?usp=sharing).
- This provided checkpoint was trained with a loss: $L_c + 10*L_s$.
- You can edit `scripts/demo.sh` file.

### Demo results
![composite](https://github.com/user-attachments/assets/38c0ad24-3e40-4fc8-aef3-484a51be257b)

## Results
I multiplied 10 with $L_s$, so the content preservation doesn't look perfect. You can change a coefficient of loss in `train.py` and find the proper value when you train the model.

#### Validation
![output_val_160](https://github.com/user-attachments/assets/c46e77ba-5e42-4cea-9e9d-bf2bbea57b36)

#### GIF
![GIF](assets/output_val_video.gif)

#### Test
![output_test_160](https://github.com/user-attachments/assets/23911566-1272-45a8-80a7-a64a44a6c6e4)

#### GIF
![GIF](assets/output_test_video.gif)

## Train
```
source scripts/train.sh
```
