import sys
import os
sys.path.append('D:/pythonProject/multi-tasks/quality-controlled-paraphrase-generation-main')
from dataclasses import dataclass, field
from typing import Optional
from datasets import GenerateMode
from data import DatasetArguments, prepare_dataset
from transformers import HfArgumentParser
from datasets import load_metric
import pandas as pd


@dataclass
class EvalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    predictions_column: Optional[str] = field(
        metadata={"help": "the source column"}
    )
    references_column: Optional[str] = field(
        default=None, metadata={"help": "the target column"}
    )
    metric_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "metric you wish to apply"}
    )
    output_path: Optional[str] = field(
        default='pairs_evals.csv', metadata={"help": "metric you wish to apply"}
    )


def main(dt, spt):

    from datasets import set_caching_enabled
    set_caching_enabled(False)

    parser = HfArgumentParser((DatasetArguments, EvalArguments))
    '''
    参数说明：
    第一个：运行代码的脚本
    --train_file: 训练数据，csv格式，两行，[source, target]
    --dataset_split: 根据原先的脚本，设置为train
    --predictions_column: 要预测的那一列数据的索引名
    --references_column: 标准参考数据的索引名
    --metric: 计算特征指标，这里如果要用成我们的独特特征需要再去研究
    --output_path: 输出文件
    '''
    sys.argv = [
        'D:/pythonProject/Chinese_Controllable_Simplification/quality-controlled-paraphrase-generation-main/QCPG/evaluate.py',
        '--train_file', f'D:/pythonProject/Chinese_Controllable_Simplification/quality-controlled-paraphrase-generation-main/data_sm/{dt}/{spt}.csv/{spt}.csv',
        '--dataset_split', 'train',
        '--predictions_column', 'source',
        '--references_column', 'target',
        '--metric', 'D:/pythonProject/Chinese_Controllable_Simplification/quality-controlled-paraphrase-generation-main/metrics/para_metric',
        '--output_path', f'D:/pythonProject/Chinese_Controllable_Simplification/quality-controlled-paraphrase-generation-main/output/{dt}/{spt}.csv/{spt}.csv'
    ]
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        dataset_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        dataset_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(os.path.abspath(os.path.dirname(eval_args.output_path)), exist_ok=True)

    dataset = prepare_dataset(dataset_args)
    column_names = dataset.column_names

    predictions = dataset[column_names[1]] if eval_args.predictions_column is None \
                  else dataset[eval_args.predictions_column]
    references = dataset[column_names[0]] if eval_args.references_column is None \
                  else dataset[eval_args.references_column]
    
    # metric = load_metric(eval_args.metric_name_or_path, experiment_id=os.getpid())
    metric = load_metric(eval_args.metric_name_or_path)
    print('Computing metric...')
    result = metric.compute(predictions=predictions, references=references)
    
    result['prediction'] = predictions
    result['reference'] = references

    try:
        df = pd.DataFrame(result)
    except:
        print(result)
        raise NotImplementedError
        
    df.to_csv(eval_args.output_path, index=False)


if __name__ == "__main__":
    # for dataset in ["parabk2", "wikians", "mscoco"]:
    #     for split in ["train", "validation", "test"]:
    #         main(dataset, split)
    main('mscoco', 'test')
