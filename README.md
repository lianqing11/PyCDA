# PyCDA
Code for Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation: A Non-Adversarial Approach. 

## Enviorment
The code is developed under the following configuration.
#### Hardware:
4-8 GPUs(With at least 11G GPU memories), which is set for the correspoinding batch size. 

#### Software:
Python(3.6) and Pytorch(0.4.1) is necessary before running the scripts. To install the required pythonn packages(expect Pytorch), run 

```
pip install -r requirements.txt
```

## Datasets

To train and validate the network, this repo use the [GTAV]() or [SYNTHIA]() as the source domain dataset and user [Cityscapes]() as the target domain dataset.

To monitor the convergence of the network, we split 500 images out of Cityscapes training dataset as our validation set and test on Cityscapes valdiation set.
You can check it in the ```./dataset/cityscapes_list/directory```.

To train on your own enviorment, please download the dataset and modify the dataset path in the corresponding cfgs docunment.
Downloaded [pretrained model](https://drive.google.com/drive/folders/1EgpKK5GwmFNM3XkyNHPtj52ZwDiogRMB?usp=sharing)


## Training

### Source only
```
sh run.sh train_source_only.py cfgs/source_only_exp001.yaml
```

### [Adabn]()
```
sh run.sh train_adabn.py cfgs/adabn_exp001.yaml
```

### [PyCDA]()
```
sh run.sh train_pycda_local.py cfgs/pycda_local_exp001.yaml
```

### Convert batchnorm statistics
```
sh run.sh test_adabn.py $your_script
```
## Performance


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


