# RSCD (BS-RSCD & JCD)

[Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes (CVPR2021)](https://arxiv.org/abs/2104.01601)

by Zhihang Zhong, Yinqiang Zheng, Imari Sato

We contributed the first real-world dataset ([BS-RSCD](https://drive.google.com/file/d/1hgzibaez7EipmPSN-3GzQO0_mlyruKGa/view?usp=sharing)) and end-to-end model (JCD) for joint rolling shutter correction and deblurring tasks.

We collected the data samples using the proposed beam-splitter acquisition system as below:  
![image](https://github.com/zzh-tech/Images/blob/master/RSCD/acquisition_system.png)

In the near future, we will add more data samples with larger distortion to the dataset...

If you are interested in real-world datasets for pure deblurring tasks, please refer to [ESTRNN&BSD](https://github.com/zzh-tech/ESTRNN).
## Prerequisites

Install the dependent packages:

```bash
conda create -n rscd python=3.8
conda activate rscd
sh install.sh
```

Download lmdb files of [BS-RSCD](https://drive.google.com/file/d/1j4gxN28KmDA7Yl1W37i87n3nFIgmZh2_/view?usp=sharing)
(or [Fastec-RS](https://drive.google.com/file/d/1JGzY_8tVVP-oy7jFL1TL84gt3yz1bry3/view?usp=sharing) for RSC tasks).

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

Please download [checkpoints](https://drive.google.com/file/d/1bGFHNjoqTGk78UTF-7qDm6hVU4Oqe7N3/view?usp=sharing) and
unzip it under the main directory.

Run a pre-trained model:

```bash
python main.py --test_only --test_checkpoint ./checkpoints/JCD_BS-RSCD.tar --video
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
