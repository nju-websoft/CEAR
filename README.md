# CEAR
Improving Continual Relation Extraction by Distinguishing Analogous Semantics, ACL 2023


Source code of the proposed model Continual Extraction for Analogous Relations (CEAR).

## Environment
Our implementation is based on Python 3.9.7 and the version of PyTorch is 1.11.0 (cuda version 11.x).  
To install PyTorch 1.11.0, you could follow the guidance on official website of [PyTorch](https://pytorch.org/).  

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
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

