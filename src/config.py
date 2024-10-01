from datasets import load_dataset, GenerateMode, Dataset
from dataclasses import dataclass, field
from typing import Optional
import json
import sys 
import unicodedata
import pandas as pd


def extract_suffix(file_name):
    compression = None
    extension = None
    parts = file_name.split('.')
    if parts[-1] in ['gz']:
        compression = parts.pop()
    if parts[-1] in ['csv', 'json']:
        extension = parts.pop()
    return extension, compression


nonprintable = (ord(c) for c in (chr(i) for i in range(sys.maxunicode)) if 'C' in unicodedata.category(c))
nonprintable_dict = {character:None for character in nonprintable}


def remove_non_printables(text):
    return text.translate(nonprintable_dict)


def bad_chars_filter(dictionary):
    result = {
        k : remove_non_printables(v) if isinstance(v, str) else v for k,v in dictionary.items()
    }
    return result


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default=None, metadata={"help": "Dataset split json in the datasets split dictionary format."}
    )
    dataset_filter: Optional[str] = field(
        default=None, metadata={"help": "python string for filtering by dataset fields, for example 'len(sentence) > 3 and sentiment < 0.5'."}
    )
    dataset_map: Optional[str] = field(
        default=None, metadata={"help": "python string for mapping dataset fields, for example 'length = len(sentence); t = 10)'."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory of the dataset cache."}
    )
    dataset_generate_mode: Optional[str] = field(
        default="reuse_dataset_if_exists", metadata={"help": "Directory of the dataset cache."}
    )
    dataset_keep_in_memory: Optional[bool] = field(
        default=False, metadata={"help": "Directory of the dataset cache."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    remove_bad_chars: Optional[bool] = field(
        default=False,
        metadata={
            "help": "remove any non printable chars from any string value. In practice remove the 'Other' category from unicodedata library."
        },
    )

    def __post_init__(self):
        self.dataset_generate_mode = GenerateMode(self.dataset_generate_mode)
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension, compression = extract_suffix(self.train_file)
                assert extension is not None, "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension, compression = extract_suffix(self.validation_file)
                assert extension is not None, "`validation_file` should be a csv or a json file."


def prepare_dataset(dataset_args, logger=None):

    if dataset_args.dataset_split is not None:
        try:
            dataset_split = eval(dataset_args.dataset_split) 
        except:
            dataset_split = str(dataset_args.dataset_split)
            if logger is not None:
                logger.warning(f"Dataset split name: '{dataset_split}' treated as string. if you want to use json make sure it can be parsed proprly.")

        if logger is not None:
            logger.info(f"Dataset is splitted by '{dataset_split}'")
    else:
        dataset_split = None
    
    if dataset_args.dataset_name is not None:
       
        datasets = load_dataset(
            dataset_args.dataset_name,
            dataset_args.dataset_config_name,
            split=dataset_split,
            keep_in_memory=dataset_args.dataset_keep_in_memory,
            cache_dir=dataset_args.dataset_cache_dir,
            download_mode=dataset_args.dataset_generate_mode,
            ignore_verifications=True,
        )
    else:
        data_files = {}
        if dataset_args.train_file is not None:
            data_files["train"] = dataset_args.train_file
            extension, compression = extract_suffix(dataset_args.train_file)
            # if compression:
            #     print(f'loading dataset from compressed file: {dataset_args.train_file}')
            #     datasets = Dataset.from_pandas(pd.read_csv(dataset_args.train_file, na_filter=False), split=dataset_args.dataset_split)
        if dataset_args.validation_file is not None:
            data_files["validation"] = dataset_args.validation_file
            extension, compression = extract_suffix(dataset_args.validation_file)
        if dataset_args.test_file is not None:
            data_files["test"] = dataset_args.test_file
            extension, compression = extract_suffix(dataset_args.test_file)

        datasets = load_dataset(extension, data_files=data_files, 
                            split=dataset_split,
                            keep_in_memory=dataset_args.dataset_keep_in_memory,
                            cache_dir=dataset_args.dataset_cache_dir,
                            na_filter=False,
                            download_mode=dataset_args.dataset_generate_mode,
                            ignore_verifications=True,
                        )

    if dataset_args.remove_bad_chars:
        datasets = datasets.map(bad_chars_filter, load_from_cache_file=False)
    
    if dataset_args.max_train_samples is not None and 'train' in datasets:
        datasets['train'] = datasets['train'].select(range(dataset_args.max_train_samples))
    
    if dataset_args.max_validation_samples is not None and 'validation' in datasets:
        datasets['validation'] = datasets['validation'].select(range(dataset_args.max_validation_samples))

    if dataset_args.max_test_samples is not None and 'test' in datasets:
        datasets['test'] = datasets['test'].select(range(dataset_args.max_test_samples))

    if dataset_args.dataset_filter:
        datasets = datasets.filter(
            lambda x: eval(dataset_args.dataset_filter, None, x),
            keep_in_memory=dataset_args.dataset_keep_in_memory,
            load_from_cache=False
        )
        if logger is not None:
            logger.info(f"Dataset was filtered by '{dataset_args.dataset_filter}'")

    if dataset_args.dataset_map:
        datasets = datasets.map(
            lambda x: eval(f"""locals() if not exec('{dataset_args.dataset_map}') else None""", None, x),
            keep_in_memory=dataset_args.dataset_keep_in_memory, 
            load_from_cache_file=False,
        )
        if logger is not None:
            logger.info(f"Dataset was mapped with '{dataset_args.dataset_filter}'")

    return datasets


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    run_tags: Optional[str] = field(
        default='',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    output_file_name: Optional[str] = field(
        default="generated_predictions.txt",
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    conditions_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The list of names of the column in the datasets containing the condition classes."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length