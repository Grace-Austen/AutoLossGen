# AutoLossGen

This repository includes the implementation for paper "AutoLossGen: Automatic Loss Function Generation for Recommender Systems".

*Zelong Li, Jianchao Ji, Yingqiang Ge, Yongfeng Zhang. 2022. [AutoLossGen: Automatic Loss Function Generation for Recommender Systems](https://arxiv.org/abs/2204.13160). In the Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)*

Original package is mainly contributed by [Zelong Li](https://github.com/lzl65825) (zelong.li@rutgers.edu), [Yongfeng Zhang](https://github.com/evison) (yongfeng.zhang@rutgers.edu).

There is an addition of a RankNet model (see src/models/RankNet.py) and alteration to src/runner/BaseRunner.py to ensure compatibility, as well as a src/run_ranknet.py file to run AutoLossGen with the RankNet model. 

Additions contributed by [Grace Austen](https://github.com/Grace-Austen). 

## Setup and Run AutoLossGen (with RankNet)

### Setup Environment
Create a new anaconda environment:

```conda env create -f environment.yml```

Activate the conda environment.

```source activate autolossgen```

Install torch using pip:

```pip install -r requirements.txt```

### Run AutoLossGen
Run an example AutoLossGen experiment using a bash script:

```. ranknet_sample.sh```

To run AutoLossGen yourself, see the following steps:
1. Navigate to AutoLossGen/src
2. Run main.py --search_loss - refer to Phase 1: Loss Search
3. Navigate to AutoLossGen/utils
4. Run reorg_log.py
5. Run loss_valid_check.py - refer to Phase 2: Validation Check
    - The log file input and output are hardcoded so ensure that they match the log file you set up in step 2.
6. Navigate to AutoLossGen/src
7. Run main.py again - refer to Phase 3: Effectiveness Test

![Image Loss](pics/Loss_Generation_Process_ver3.0.jpg)  

#### Phase 1: Loss Search
```
python main.py --epoch 10000 --child_num_branches 9 --child_num_layers 10 --search_loss --gpu 0 --sample_branch_id --sample_skip_id --controller_num_aggregate 10 --controller_train_steps 10 --log_file ../log/log_0.txt --formula_path ../model/Formula_0.txt --train_with_optim --dataset ml100k01-1-5 --model_name BiasedMF --random_seed 42
```

Note: 
- If you're searching for a loss function for RankNet, use run_ranknet.py instead of main.
- If the number of epochs is set too small, the performance may not good enough as the search is not throughout. Li et al recommend a large number of epochs, and you can stop it manually. 
- search_loss indicates the model is in loss searching mode
- dataset and model_name are used to control what dataset is used to train each iteration of the model model_name when finding loss functions

#### Phase 2: Validation Check
```
python loss_valid_check.py
```

The hardcoded file paths are as follows:
- input log file = log/log_0.csv
- output log file = log/check_log_0.csv

#### Phase 3: Effectiveness Test
```
python main.py --epoch 10000 --log_file ../log/log_1.txt --model_name BiasedMF --loss_func MaxR --metric AUC --dataset ml100k01-1-5 --smooth_coef 1e-6
```

Note: Here the program is determining the effectiveness of the loss function, hence loss_func needs to be specified and search_loss is omitted. 

#### Troubleshooting
If you're having issues with json.decoder e.g.:

```json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 1)```

Then remove the dataset/[dataset_you used]/[dataset_you used].info.json file, or otherwise move it from the directory.

I would also advise you remove the dataset/[dataset_you used]/[dataset_you used].train_group.csv and the dataset/[dataset_you used]/[dataset_you used].vt_group.csv

## Environments

Python 3.10.13

Necessary packages:

```
torch==1.13.0
numpy==1.26.0
scikit-learn==1.2.2
scipy==1.11.3
pandas==2.1.1
tqdm==4.65.0
```

## Reference

- Li et al leveraged the dataset of [NCR](https://github.com/rutgerswiselab/NCR) projects to implement our experiment.
- Li et al implemented our Controller part referring to [this project](https://github.com/TDeVries/enas_pytorch/), which is a PyTorch implementation of paper [Efficient Neural Architecture Search via Parameters Sharing](https://arxiv.org/abs/1802.03268).
- I based my implementation of RankNet off of [RankNet-Pytorch](https://github.com/yanshanjing/RankNet-Pytorch/).
- I based my RankNet loss function off of the implementation in [this project](https://github.com/allegro/allRank/) which supported the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084).