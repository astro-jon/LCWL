# LCWL

🤗Code and pretrained models for reproducing experiments in "Label Confidence Weighted Learning for Target-Level Sentence Simplification".

## How to Use
## [data](data)
- [newsela](data%2Fnewsela): owing to licence, you need to request the raw data and then place it into data/newsela
- [paranmt](data%2Fparanmt): the parapohrase which has been labeled and recorded the classifier confidence

## our code: 
run the bert-based classifier to label the paraphrase
```python
python readability_predict/src/main.py
```
run the text2text generation model:
```
python src/train_201.py
```

## Citation

If you found this repository useful, please consider

```latex
@inproceedings{qiu-zhang-2024-label,
    title = Label Confidence Weighted Learning for Target-level Sentence Simplification,
    author = Qiu, Xin Ying  and Zhang, Jingshen,
    booktitle = Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
    month = nov,
    year = 2024,
    address = Miami, Florida, USA,
    publisher = Association for Computational Linguistics,
    url = https://aclanthology.org/2024.emnlp-main.999/,
    doi = 10.18653/v1/2024.emnlp-main.999,
    pages = 18004--18019,
}
```




