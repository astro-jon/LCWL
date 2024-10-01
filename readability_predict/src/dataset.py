import torch
from torch.utils.data import Dataset


class InputDataset(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # sentence_id = self.data['sentence_id'][item]
        sentence = self.data['sentence'][item]
        # sentence = sentence.replace("[SEP]", "</s>")
        label = self.data['label'][item]
        # features = [round(i, 2) for i in eval(self.data['9features'][item])]
        # features = ''.join(f'<cond_{num}_{spe}>'.upper() for num, spe in enumerate(features))
        # sentence = sentence + '[SEP]' + features
        label = torch.tensor(label, dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            # 'sentence_id': sentence_id,
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': label,
        }


class InputDatasetAddOpt(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_id = self.data['item_id'][item]
        text = ' '.join(eval(self.data['article'][item]))
        options = eval(self.data['options'][item])
        dis = ' '.join([k for i in options for j in i for k in j])

        label = self.data['level'][item]
        label = torch.tensor(label, dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            text,
            dis,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'item_id': item_id,
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': label
        }





