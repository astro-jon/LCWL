# LCWL

🤗Code and pretrained models for reproducing experiments in "Label Confidence Weighted Learning for Target-Level Sentence Simplification".

## How to Use
## [data](data)
- [newsela](data%2Fnewsela): owing to licence, you need to request the raw data and then place it into data/newsela
- [paranmt](data%2Fparanmt): the parapohrase which has been labeled and recorded the classifier confidence

## We will continuely update our code...... 
run the bert-based classifier to label the paraphrase
```python
python readability_predict/src/main.py
```
run the text2text generation model:
```
python src/train_201.py
```

If you have any questiones, or there exists some bugs on running codes, please email us: `audbut0702@163.com`




