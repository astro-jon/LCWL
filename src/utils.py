import pandas as pd


def read_hsk_vocab(path):
    hsk_vocab = {}
    vocab = pd.read_excel(path)
    level_keys = {
        '一级': 1,
        '二级': 2,
        '三级': 3,
        '四级': 4,
        '五级': 5,
        '六级': 6,
        '七-九级': 7
    }
    for k in level_keys:
        for word in vocab[k].dropna():
            if '/' in word:
                words = word.split('/')
                for w in words:
                    hsk_vocab[w.strip()] = level_keys[k]
            else:
                hsk_vocab[word.strip()] = level_keys[k]
    return hsk_vocab


def count_nums(sys_sentences, hsk_vocab):
    level_freq = {}
    num_tokens = 0
    num_types = 0
    for seg_line in sys_sentences:
        for word in seg_line:
            if word in hsk_vocab:
                level = hsk_vocab[word]
            else:
                level = 8
            if level not in level_freq:
                level_freq[level] = {}
            if word not in level_freq[level]:
                level_freq[level][word] = 0
                num_types += 1
            level_freq[level][word] += 1
            num_tokens += 1
    return level_freq, num_tokens, num_types