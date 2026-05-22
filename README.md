# Code for the NeurIPS 2024 paper ``Cross-Modality Perturbation Synergy Attack for Person Re-identification".



## [Paper](paper/CMPS.pdf) 

## Requirements:
* python 3.7
* CUDA==11.2
* faiss-gpu==1.6.0


## Preparing Data

* There is a processed tar file in [BaiduYun](https://pan.baidu.com/s/160oRNcDSemBprqBUBX0PUQ?pwd=9pmk) (Password: 9pmk)  with all needed files.


## Run our code
 
See run.sh for more information.

If you find this code useful in your research, please cite:

```
@article{gong2024cross,
	title={Cross-modality perturbation synergy attack for person re-identification},
	author={Gong, Yunpeng and Zhong, Zhun and Qu, Yansong and Luo, Zhiming and Ji, Rongrong and Jiang, Min},
	journal={Advances in Neural Information Processing Systems},
	volume={37},
	pages={23352--23377},
	year={2024}
}
```

## Acknowledgments

The code is based on [LTA](https://github.com/finger-monkey/LTA_and_joint-defence), [Random Color Erasing](https://github.com/finger-monkey/Data-Augmentation) and [Mata_attack](https://github.com/WJJLL/Meta-Attack-Defense). 

Recently, some users have reported that running the same code now produces results that differ significantly from those obtained a few months ago. We suspect this is most likely caused by updates to packages or the runtime environment. At the moment, there is no good solution, and the author does not have the time or capacity to maintain it. Please make sure that your environment and package versions are configured according to the versions used when the [Mata_attack](https://github.com/WJJLL/Meta-Attack-Defense) code was released.

![Uploading image.png…]()


If you use the code, please cite their paper.
```
@inproceedings{colorAttack2022,
  title={Person re-identification method based on color attack and joint defence},
  author={Gong, Yunpeng and Huang, Liqing and Chen, Lifei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4313--4322},
  year={2022}
}
```
```
@article{gong2021eliminate,
  title={Eliminate deviation with deviation for data augmentation and a general multi-modal data learning method},
  author={Gong, Yunpeng and Huang, Liqing and Chen, Lifei},
  journal={arXiv preprint arXiv:2101.08533},
  year={2021}
}
```






## Contact Me

Email: fmonkey625@gmail.com





<a href="https://info.flagcounter.com/mRXd"><img src="https://s11.flagcounter.com/count/mRXd/bg_3F90EB/txt_FFFFFF/border_CCCCCC/columns_8/maxflags_12/viewers_Visitors+of+HMC+repo/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
