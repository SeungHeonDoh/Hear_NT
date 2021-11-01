# HEAR_NT

Submission for HEAR2021 workshop @ NeurIPS'21.

Multi-head supervised audio representations with leaf-audio.
 

### Important module

This development environment is configured according to the Docker below.
- https://github.com/neuralaudio/hear-eval-kit/blob/main/docker/Dockerfile-cuda11.2

==================================================================

- python : 3.7 (apt)
- baseline docker : nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

==================================================================


### Setup lib

```
pip3 install git+https://github.com/SeungHeonDoh/Hear_NT.git
```

or

```
git clone https://github.com/SeungHeonDoh/Hear_NT.git
cd Hear_NT
pip3 install -e .
```

### Setup pretrained_model

You may also download them from drive: [hear_nt_pretrained](https://drive.google.com/file/d/1Lo7xx7wW_tTpxagK6R9rIMVAbSG4G--P/view?usp=sharing)


### Model

|   Model Name    | Sampling Rate | Embedding Size |  Location  |
| --------------- | ------------- | -------------- |  --------  |
|     HEAR_NT     |    22050      |      4096     |  [hear_nt_pretrained](https://drive.google.com/file/d/1Lo7xx7wW_tTpxagK6R9rIMVAbSG4G--P/view?usp=sharing) |


**reference**
- https://github.com/google-research/leaf-audio 

```
@article{zeghidour2021leaf,
  title={LEAF: A Learnable Frontend for Audio Classification},
  author={Zeghidour, Neil and Teboul, Olivier and de Chaumont Quitry, F{\'e}lix and Tagliasacchi, Marco},
  journal={ICLR},
  year={2021}
}
```
