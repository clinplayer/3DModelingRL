# Modeling 3D Shapes by Reinforcement Learning

We made an initial attempt to model 3D shapes like human modelers using deep reinforcement learning (DRL). This repository contains the source code for the paper [Modeling 3D Shapes by Reinforcement Learning](https://arxiv.org/abs/2003.12397).

<a href="https://arxiv.org/abs/2003.12397">
<img src="imgs/teaser.jpg" style="width:400px; display: block; margin-left: auto; margin-right: auto;"/>
</a>

## Code
### Installation
You need to install [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/). This code is tested under Python 3.7.4, PyTorch 1.3.0, NumPy 1.17.2 and SciPy 1.3.1 on Ubuntu 18.04.4.

The repository contains a part of the code from [binvox](https://www.patrickmin.com/binvox/).

### Training
* Train Prim-Agent first
```
cd Prim-Agent
python train.py
```
* Then use the trained Prim-Agent to generate primitives and edge loop files for all the data
```
python generate_edgeloop.py
```
* Train Mesh-Agent using the output of Prim-Agent
```
cd Mesh-Agent
python train.py
```
* Will need to provide paths to the training data and saving results & logs when calling
* Can change the setting by modifying the parameters in `Prim-Agent/config.py` or `Mesh-Agent/config.py` 

### Testing
* Call `Prim-Agent/test.py` and `Mesh-Agent/test.py` for testing. Will need to provide paths to the data and the pre-trained model when calling.

### Download
* Data [data.zip](https://drive.google.com/file/d/1inwGXugUEB_vbmTjl33gfWWPhAw594Fv/view?usp=sharing)
* Pre-trained model [pretrained.zip](https://drive.google.com/file/d/1VTM4--sf0xas29s_frF7_tZsPNFTNcFL/view?usp=sharing)
* Unzip the downloaded files and use them to replace `data` and `pretrained` folders; then you can directly run the code without modifying the arguments when calling `train.py` and `test.py`


## Fast demo

## Citation:  
If you find our work useful in your research, please consider citing:
```
@article{lin2020modeling,
  title={Modeling 3D Shapes by Reinforcement Learning},
  author={Lin, Cheng and Fan, Tingxiang and Wang, Wenping and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2003.12397},
  year={2020}
}
```

## Contact:
If you have any questions, please email Cheng Lin at chlin@hku.hk.
