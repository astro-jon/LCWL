import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDataset, InputDatasetAddOpt
from model import BertForSeq
from utils import log_creater, seed_everything
from transformers.utils.notebook import format_time
import argparse
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

os.environ["WANDB_DISABLED"] = "true"
TEST_DATASET_PATH = "../data/paraphrase_forpred.csv"
PRED_SAVE_PATH = "../data/paraphrase_label_preded.csv"


def train(model, train_loader, val_loader, test_loader, log):
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = AdamW([{'params': weight_params, 'weight_decay': 1e-5},
                       {'params': bias_params, 'weight_decay': 0}],
                      lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    best_model_path = args.save_path + '/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    log.info("Train batch size = {}".format(args.batch_size))
    log.info("Total steps = {}".format(total_steps))
    log.info("Training Start!")
    log.info('')

    best_s, step_count, best_dict = -1, 0, {}
    for epoch in range(args.epochs):
        total_train_loss = 0
        t0 = time.time()
        model.to(args.device)
        model.train()
        for step, batch in enumerate(train_loader):
            step_count += 1
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['label'].to(args.device)
            model.zero_grad()
            loss, output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('step : {},   loss : {}'.format(step, loss.item()))
        avg_train_loss = total_train_loss / len(train_loader)
        train_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch, args.epochs, avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')
        acc_list, avg_acc, f1_s = evaluate(model, val_loader)
        tmp_s = avg_acc * 0.5 + f1_s * 0.5
        val_time = format_time(time.time() - t0)
        acc_list = [round(n, 4) for n in acc_list]
        if tmp_s > best_s:
            best_s = tmp_s
            torch.save(model.state_dict(), best_model_path + 'model.pt')
            best_dict = {"acc": avg_acc, "f1": f1_s}
            print('Model Saved!')
        log.info('====Epoch:[{}/{}] Avg_List = {} | Avg_Acc = {:.5f} | F1_score = {:.5f} | \n Best_score = {:.5f} | {} ===='.format(
            epoch, args.epochs, acc_list, avg_acc, f1_s, best_s, best_dict))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

    log.info('   Training Completed!')
    log.info('')
    model.load_state_dict(torch.load(best_model_path + 'model.pt'))
    predict_label(model, test_loader)


def predict_label(model, data_loader):
    model.eval()
    label_list, pred_result = [], []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            loss, output = model(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids,
                                 labels = labels)

        output = torch.argmax(output, dim = 1)
        pred_result += output.data.tolist()
    test_df = pd.read_csv(TEST_DATASET_PATH)
    test_df["pred_label"] = pred_result
    test_df.to_csv(PRED_SAVE_PATH, sep = ',', index = False)


def evaluate(model, data_loader):
    model.eval()
    num_data = 0
    label_list, pred_result = [], []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            loss, output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        output = torch.argmax(output, dim=1)
        pred_result += output.data.tolist()
        label_list += labels.data.tolist()
        num_data += len(labels)

    index0_list = [idx for idx, i in enumerate(label_list) if i == 0]
    index1_list = [idx for idx, i in enumerate(label_list) if i == 1]
    index2_list = [idx for idx, i in enumerate(label_list) if i == 2]
    index3_list = [idx for idx, i in enumerate(label_list) if i == 3]
    index4_list = [idx for idx, i in enumerate(label_list) if i == 4]
    label_list0, pred_list0 = np.array(label_list)[index0_list], np.array(pred_result)[index0_list]
    label_list1, pred_list1 = np.array(label_list)[index1_list], np.array(pred_result)[index1_list]
    label_list2, pred_list2 = np.array(label_list)[index2_list], np.array(pred_result)[index2_list]
    label_list3, pred_list3 = np.array(label_list)[index3_list], np.array(pred_result)[index3_list]
    label_list4, pred_list4 = np.array(label_list)[index4_list], np.array(pred_result)[index4_list]
    acc0 = accuracy_score(label_list0, pred_list0)
    acc1 = accuracy_score(label_list1, pred_list1)
    acc2 = accuracy_score(label_list2, pred_list2)
    acc3 = accuracy_score(label_list3, pred_list3)
    acc4 = accuracy_score(label_list4, pred_list4)
    f1 = f1_score(label_list, pred_result, average="macro")
    acc_list = [acc0, acc1, acc2, acc3, acc4]
    avg_acc = round(np.mean(acc_list), 3)

    return acc_list, avg_acc, f1


def main():
    seed_everything(args.seed)

    log = log_creater(output_dir='../cache/logs/')
    log.info(args.model_path)
    log.info('EPOCH = {}; LR = {}'.format(args.epochs, args.learning_rate))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tr_data = pd.read_csv("../data/newsela_train_forpred.csv")
    va_data = pd.read_csv("../data/newsela_dev_forpred.csv")
    test_data = pd.read_csv(TEST_DATASET_PATH)

    tr_dataset = InputDataset(tr_data, tokenizer, args.max_input_length)
    tr_data_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    va_dataset = InputDataset(va_data, tokenizer, args.max_input_length)
    va_data_loader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False)
    te_dataset = InputDataset(test_data, tokenizer, args.max_input_length)
    te_data_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)

    model = BertForSeq(args, tokenizer)

    train(model, tr_data_loader, va_data_loader, te_data_loader, log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '../best_model/model_saved')
    parser.add_argument('--model_path', type = str, default = 'bert-base-uncased')
    parser.add_argument('--num_labels', type = int, default = 5)  # 类别数量
    parser.add_argument('--max_input_length', type = int, default = 256)
    parser.add_argument('--epochs', type = int, default = 80)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--num_feature', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 2023)
    parser.add_argument('--device', type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    main()














