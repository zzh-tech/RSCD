# RSCD (BS-RSCD & JCD)

**[CVPR2021]** [Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes](https://arxiv.org/abs/2104.01601) ([CVF Link](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Towards_Rolling_Shutter_Correction_and_Deblurring_in_Dynamic_Scenes_CVPR_2021_paper.pdf))

by [Zhihang Zhong](https://zzh-tech.github.io/), Yinqiang Zheng, Imari Sato

We contributed the first real-world dataset ([BS-RSCD](https://drive.google.com/file/d/1DZe1wUgBs_ORS2O4_zxjqNnYoaKwwNSQ/view?usp=sharing)) and end-to-end model (JCD) for joint rolling shutter correction and deblurring tasks.

We collected the data samples using the proposed beam-splitter acquisition system as below:

![image](https://drive.google.com/uc?export=view&id=1sIaHIz-LlqqM0RMzl78Ub_1WD4ozPGUw)

In the near future, we will add more data samples with larger distortion to the dataset ...

If you are interested in real-world datasets for pure deblurring tasks, please refer to [ESTRNN & BSD](https://github.com/zzh-tech/ESTRNN).
## Prerequisites

Install the dependent packages:

```bash
conda create -n rscd python=3.8
conda activate rscd
sh install.sh
```

Download lmdb files of [BS-RSCD](https://drive.google.com/file/d/1NmAbkpL2IWRdgZ23aIMYWiivGt24Rer8/view?usp=sharing)
(or [Fastec-RS](https://drive.google.com/file/d/1vs9sxav9h9Yjjs11KCxKtFuzH2gaRZAx/view?usp=sharing) for RSC tasks).

(PS, for how to create lmdb file, you can refer to ./data/create_rscd_lmdb.ipynb)
## Training

Please specify the *\<path\>* (e.g. "./dataset/ ") where you put the dataset file or change the default value in "
./para/paramter.py".

Train JCD on BS-RSCD:

```bash
python main.py --data_root <path> --model JCD --dataset rscd_lmdb --video
```

Train JCD on Fastec-RS:

```bash
python main.py --data_root <path> --model JCD --dataset fastec_rs_lmdb --video
```

## Testing

Please download [checkpoints](https://drive.google.com/file/d/1_B6Q9K9V1WycKkEQscS5Hhd6WxR3GRT9/view?usp=sharing) and
unzip it under the main directory.

Run the pre-trained model on BS-RSCD:

```bash
python main.py --test_only --dataset rscd_lmdb --test_checkpoint ./checkpoints/JCD_BS-RSCD.tar --video
```

Inference for video file:
```bash
python video_inference.py --src <input_path> --dst <output_path> --checkpoint ./checkpoints/JCD_BS-RSCD.tar
```

## Citing

If BS-RSCD and JCD are useful for your research, please consider citing:

```bibtex
@InProceedings{Zhong_2021_Towards,
  title={Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes},
  author={Zhong, Zhihang and Zheng, Yinqiang and Sato, Imari},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year={2021}
}
```
