# Code for NeurIPS 2024 paper ``Cross-Modality Perturbation Synergy Attack for Person Re-identification".




## [Paper](pdfs/XXXX.pdf)

## Requirements:
* python 3.7
* CUDA==10.1
* faiss-gpu==1.6.0
* Other necessary packages listed in [requirements.txt](requirements.txt)

## Preparing Data

* There is a processed tar file in [BaiduYun](https://pan.baidu.com/s/160oRNcDSemBprqBUBX0PUQ?pwd=9pmk) (Password: 9pmk)  with all needed files.

## Preparing Models

* Download re-ID models and dataset from [BaiduYun](https://pan.baidu.com/s/1LU2EYmLRGen49F3FgcXvZQ?pwd=z7um) (Password: z7um)


## Run our code
 
See run.sh for more information.

If you find this code useful in your research, please consider citing:

```
@article{gong2024cross,
  title={Cross-Modality Perturbation Synergy Attack for Person Re-identification},
  author={Gong, Yunpeng and others},
  journal={arXiv preprint arXiv:2401.10090},
  year={2024}
}
```

## Acknowledgments

Our code is based on [Random Color Erasing](https://github.com/finger-monkey/Data-Augmentation),[UAP-Retrieval](https://github.com/theFool32/UAP_retrieval) and [LTA](https://github.com/finger-monkey/LTA_and_joint-defence)  
if you use our code, please also cite their paper.
```
@misc{RCE2024,
      title={Exploring Color Invariance through Image-Level Ensemble Learning}, 
      author={Yunpeng Gong and Jiaquan Li and Lifei Chen and Min Jiang},
      year={2024},
      eprint={2401.10512},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@inproceedings{Li_2019_ICCV,
    author = {Li, Jie and Ji, Rongrong and Liu, Hong and Hong, Xiaopeng and Gao, Yue and Tian, Qi},
    title = {Universal Perturbation Attack Against Image Retrieval},
    booktitle = {ICCV},
    year = {2019}
}
```
```
@inproceedings{colorAttack2022,
  title={Person re-identification method based on color attack and joint defence},
  author={Gong, Yunpeng and Huang, Liqing and Chen, Lifei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4313--4322},
  year={2022}
}
```



## Contact Me

Email: fmonkey625@gmail.com

