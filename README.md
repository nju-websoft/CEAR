# CEAR
Source code of the proposed model Continual Extraction for Analogous Relations (CEAR) in ACL 2023 long paper [Improving Continual Relation Extraction by Distinguishing Analogous Semantics](https://arxiv.org/abs/2305.06620).

> Continual relation extraction (RE) aims to learn constantly emerging relations while avoiding forgetting the learned relations. Existing works store a small number of typical samples to re-train the model for alleviating forgetting. However, repeatedly replaying these samples may cause the overfitting problem. We conduct an empirical study on existing works and observe that their performance is severely affected by analogous relations. To address this issue, we propose a novel continual extraction model for analogous relations. Specifically, we design memory-insensitive relation prototypes and memory augmentation to overcome the overfitting problem. We also introduce integrated training and focal knowledge distillation to enhance the performance on analogous relations. Experimental results show the superiority of our model and demonstrate its effectiveness in distinguishing analogous relations and overcoming overfitting.

## Environment
Our implementation is based on Python 3.9.7 and the version of PyTorch is 1.11.0 (cuda version 11.x).  
To install PyTorch 1.11.0, you could follow the official guidance of [PyTorch](https://pytorch.org/).  

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then, other dependencies could be installed by running:
```
pip install -r requirements.txt
```

## Dataset
We use two datasets in our experiments, FewRel and TACRED.

The splited datasets and task orders could be found in the corresponding directory of `data/`.


## Usage

To reproduce the results of main experiment:
```
bash FewRel.sh
bash tacred.sh
```

We conduct all experiments on a single RTX A6000 GPU with 48GB memory.


## Citation
```bibtex
@inproceedings{zhao-2023-improving,
  author = {Wenzheng Zhao and
            Yuanning Cui and
            Wei Hu},
  title = {Improving Continual Relation Extraction by Distinguishing Analogous Semantics},
  booktitle = {ACL},
  year = {2023},
}
```
