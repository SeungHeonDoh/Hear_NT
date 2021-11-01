# HEAR_NT

Submission for HEAR2021 workshop @ NeurIPS'21.
Multi-head supervised audio representations with leaf-audio.
 
**reference**
- https://github.com/google-research/leaf-audio 



## Model

This development environment is configured according to the Docker below.
- https://github.com/neuralaudio/hear-eval-kit/blob/main/docker/Dockerfile-cuda11.2

==================================================================

Important module
------------------------------------------------------------------
- python : 3.7 (apt)
- baseline docker : nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

==================================================================


### Setup lib

```
pip3 install git+https://github.com/SeungHeonDoh/Hear_NT.git
```

```
git clone https://github.com/SeungHeonDoh/Hear_NT.git
cd Hear_NT
pip3 install -e .
```

### Setup pretrained_model

You may also download them from drive: [hear_nt_pretrained](https://drive.google.com/file/d/1cwBUp-DlNzAa_b76jxx9TJNmy24-BlGX/view?usp=sharing)

### Setup dataset

for the following tasks with 22050 sample rate:

    dcase2016_task2-hear2021-full
    nsynth_pitch-v2.2.3-5h
    nsynth_pitch-v2.2.3-50h
    speech_commands-v0.0.2-5h
    speech_commands-v0.0.2-full

```
cd script
bash download_dataset.sh
```

### Evaluation


```
time python3 -m heareval.embeddings.runner heartemplate --model ../hear_model/model/20210925_000306_inception_Mixed_segment1-done/best.h5 --tasks-dir hear-2021.0.3/tasks/
```