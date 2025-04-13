import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score

# target_tags = [
#     'implementation', 'math', 'greedy', 'dp', 'datastructures', 'constructivealgorithms',
#     'bruteforce', 'graphs', 'binarysearch', 'sortings', 'dfsandsimilar', 'trees', 'strings',
#     'numbertheory', 'combinatorics', 'twopointers', 'bitmasks', 'geometry', 'dsu', 'shortestpaths',
#     'probabilities', 'divideandconquer', 'games', 'hashing', 'interactive', 'flows'
# ]

target_tags = [
    'math', 'greedy', 'dp', 'datastructures', 'constructivealgorithms',
    'graphs', 'trees', 'binarysearch', 'sortings', 'strings', 'twopointers', 'bitmasks', 'geometry',
    'divideandconquer', 'games', 'hashing', 'interactive', 'flows'
]

df = pd.read_csv('data.csv')

def process_tags(tag_str):
    if pd.isna(tag_str):
        return []
    tags = [token.strip() for token in tag_str.split(',')]
    filtered = [tag for tag in tags if not re.match(r'^\*\d+$', tag)]
    return filtered

df['processed_tags'] = df['problem_tags'].apply(process_tags)
def create_label_vector(tag_list):
    for i in range(len(tag_list)):
        if tag_list[i] in ['dfsandsimilar', 'dsu', 'shortestpaths']:
            tag_list[i] = 'graphs'
        if tag_list[i] in ['numbertheory', 'combinatorics', 'probabilities', 'matrices', 'fft', 'chineseremaindertheorem']:
            tag_list[i] = 'math'
    return [1 if tag in tag_list else 0 for tag in target_tags]

df['label_vector'] = df['processed_tags'].apply(create_label_vector)
df = df[df['label_vector'].apply(lambda x: 1 in x)]

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 512

lengths = df['problem_statement'].apply(lambda x: len(tokenizer.tokenize(x)))
print(lengths.describe())
def predict_problem(problem_statement, model, tokenizer, threshold=0.5, max_length=512):
    # Tokenize
    encoding = tokenizer(
        problem_statement,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encoding = {key: val.to(model.device) for key, val in encoding.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits.cpu().numpy()[0]
    
    # Sigmoid activation
    probs = 1 / (1 + np.exp(-logits))
    
    # Thresholding
    pred_labels = [target_tags[i] for i, p in enumerate(probs) if p > threshold]
    
    return pred_labels, probs

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # 注意这里 texts 是 Series 类型，所以用 .iloc[idx]
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 去掉 batch 维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

train_dataset = MultiLabelDataset(train_df['problem_statement'], train_df['label_vector'], tokenizer, max_length)
val_dataset = MultiLabelDataset(val_df['problem_statement'], val_df['label_vector'], tokenizer, max_length)
print(len(train_dataset), len(val_dataset))
# print(train_dataset[0])
# print(train_dataset[1])
num_labels = len(target_tags)
checkpoint_path = './results18/checkpoint-6000'
model = BertForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.config.problem_type = "multi_label_classification"

def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    # 使用 Sigmoid 激活
    probs = 1 / (1 + np.exp(-logits))
    # 阈值设为 0.5
    preds = (probs > 0.5).astype(int)
    micro_f1 = f1_score(labels, preds, average='micro')
    return {"micro_f1": micro_f1}

training_args = TrainingArguments(
    output_dir='./results18',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy='steps',
    save_strategy='steps',
    save_steps=200,
    logging_steps=200,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='micro_f1',
    logging_dir='./logs',           # 指定 TensorBoard 日志的保存路径
    report_to=['tensorboard']       # 指定将日志报告给 TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

predict_problem_statement = """
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

"""

predicted_tags, predicted_probs = predict_problem(predict_problem_statement, model, tokenizer, threshold=0.3)
print(predicted_tags, predicted_probs)
# trainer.train()
# trainer.train(resume_from_checkpoint=checkpoint_path)
# val_result = trainer.predict(val_dataset)
# logits = val_result.predictions          # 形状 [num_samples, num_labels]
# true_labels = val_result.label_ids       # 形状 [num_samples, num_labels]

# probs = 1 / (1 + np.exp(-logits))  


# best_thr = 0.0
# best_f1 = 0.0

# pred = (probs > 0.5).astype(int)
# for i in range(10):
#     print(pred[i], true_labels[i])
# for thr in np.arange(0.0, 1.01, 0.01):
#     preds = (probs > thr).astype(int)
#     f1 = f1_score(true_labels, preds, average='micro')
#     if f1 > best_f1:
#         best_f1 = f1
#         best_thr = thr

# print("验证集中找到的最优阈值: {:.2f}".format(best_thr))
# print("对应的 micro F1: {:.4f}".format(best_f1))

# # --------------------
# # 5. 对新数据（如训练集 / 测试集）进行预测并应用该阈值
# # --------------------
# #   如果你想在训练集上查看效果：
# train_result = trainer.predict(train_dataset)
# train_logits = train_result.predictions
# train_probs = 1 / (1 + np.exp(-train_logits))

# #   用搜索到的 best_thr 做最终预测
# train_preds = (train_probs > best_thr).astype(int)

# #   也可以查看训练集上的指标：
# train_f1 = f1_score(train_result.label_ids, train_preds, average='micro')
# print("训练集上的 F1: {:.4f}".format(train_f1))