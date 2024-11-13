# sapiens-api

Thanks to Facebook research team who has created and open sourced Sapiens - Foundational human vision models. You can find their repository here: https://github.com/facebookresearch/sapiens.git. 

This library is intended to create a wrapper library and offer Sapiens as API service.

Instruction to run
```
conda create -n sapiens_api python=3.10
conda activate sapiens_api
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

I am using the following version.
```
pytorch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
```