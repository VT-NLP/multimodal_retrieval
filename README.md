# multimodal_retrieval



## Dataset
Put the kmass dataset in the data folder. The expected folder structure is like data/kmass/documents, data/kmass/queryio/in and data/kmass/queryio/out.

## Run
### 1. Create conda environment and install requirements

 
```console
conda create -n multimodal_retrieval -y python  && conda activate multimodal_retrieval
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Run

```console
python multimodal_retrieval.py
```

### 3. Compute Precision Score
```console
PYTHONPATH=. python data_util/kmass_data_util.py
```


 
 
## Credit: 

This code is from [End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models](https://github.com/VT-NLP/Mocheg) codebase. 

Please use the following citation:
```
@article{yao2022end,
  title={End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models},
  author={Yao, Barry Menglong and Shah, Aditya and Sun, Lichao and Cho, Jin-Hee and Huang, Lifu},
  journal={arXiv preprint arXiv:2205.12487},
  year={2022}
}

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```