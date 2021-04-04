# RSCD (BS-RSCD & JCD)

[Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes (CVPR2021)](https://drive.google.com/file/d/1fsxMAQrt_1bZYXlRxC4xQL2K9EejO4DQ/view?usp=sharing)

by Zhihang Zhong, Yinqiang Zheng, Imari Sato

We contributed the first real-world
dataset ([BS-RSCD](https://drive.google.com/file/d/1hgzibaez7EipmPSN-3GzQO0_mlyruKGa/view?usp=sharing)) and end-to-end model (JCD) for joint rolling
shutter correction and deblurring task.

We collected the data samples using the proposed beam-splitter acquisition system as below:  
![image](https://github.com/zzh-tech/Images/blob/master/RSCD/acquisition_system.png)

In the near future, we will add more data samples with larger distortion to the dataset...

## Prerequisites

Install the dependent packages:

```bash
conda create -n rscd python=3.8
conda activate rscd
sh install.sh
```

Download lmdb file of [BS-RSCD](https://drive.google.com/file/d/1j4gxN28KmDA7Yl1W37i87n3nFIgmZh2_/view?usp=sharing)
(or [Fastec-RS](https://drive.google.com/file/d/1ZHFi6SrftR-t57vvqneG4jlU3_v4Jh84/view?usp=sharing) for RSC task).

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

If JCD and BS-RSCD are useful for your research, please consider citing:

```bibtex
@InProceedings{Zhong_2021_Towards,
  title={Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes},
  author={Zhong, Zhihang and Zheng, Yinqiang and Sato, Imari},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year={2021}
}
```
