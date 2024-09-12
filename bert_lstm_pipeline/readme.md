### 基于Bert+LSTM+TextCNN的电商评论分类系统





### 项目要求
- 电商评论分类：好评/差评
- 训练集/验证集划分
- 数据分析：正负样本数，文本平均长度等
- 实验对比3种以上模型结构的分类效果
- 每种模型对比模型预测速度
- 总结成表格输出

### 项目文件内容
- model.py
```markdown
根据config配置中的`model_type`字段，来选择到底使用哪个模型

可选的模型包括
- Fast-Text
- LSTM
- GRU
- RNN
- CNN
- GatedCNN
- StackedGatedCNN
- RCNN
- Bert
- Bert+LSTM
- Bert+CNN

```

- config.py
```python
# 设置了Config字典用来存储各种训练参数
Config = {
    "model_path": "output",
    "train_data_path": "./data/train_tag_news.json",
    "valid_data_path": "./data/valid_tag_news.json",
    "vocab_path":"nn_pipline\chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\pre-trained-models\bert-base-chinese",
    "seed": 987
}
```

- loader.py
```markdown
自定义的Dataset类，用于加载词表，加载电商评论（使用bertTokenizer进行分词），根据前面两者来创建训练集

创建DataLoader为训练做准备
```

- main.py
```markdown
 - 主训练流程, 分多个batch进行训练

```

- evaluate.py
```markdown
使用验证集进行模型性能评估，并记录每个模型的关键性能指标
```

  
### 项目细节
1. model.py中，我们实现了Bert， LSTM, Bert+LSTM, TextRCNN 等文本分类模型， 并且，我们根据config文件中“model”字段的配置，自动选择加载某个模型进行训练.

```python
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

```


## 如何运行
- 打开main.py运行
![image](https://github.com/user-attachments/assets/4d276d0e-3ea1-4dd1-9805-d240229a00d0)

- 所有模型的分类性能对比文件会导出在`model_output.csv`


### 请自行修改config文件中所有的路径！！！
![image](https://github.com/user-attachments/assets/43525d94-6ae5-4356-9863-1da2a33ed64c)



## 实验结果
![image](https://github.com/user-attachments/assets/6c68f3a4-d611-4c30-8ae7-9df94524fd29)
![image](https://github.com/user-attachments/assets/576ea2cd-52a8-44e5-bac9-fd931bd987ba)
![image](https://github.com/user-attachments/assets/5eeb59ff-c92c-454f-9165-ca2387cde757)

- 模型分类性能结果
```markdown

| model_type        | acc                  | time                 | sample_len |  
|-------------------|----------------------|----------------------|------------|  
| fast_text         | 0.8723404255319149   | 0.13781332969665527  | 30.0       |  
| lstm              | 0.9148936170212766   | 0.19060349464416504  | 30.0       |  
| gru               | 0.925531914893617    | 0.20266389846801758  | 30.0       |  
| rnn               | 0.9148936170212766   | 0.20966029167175293  | 30.0       |  
| cnn               | 0.9574468085106383   | 0.16521191596984863  | 30.0       |  
| gated_cnn         | 0.9361702127659575   | 0.18055129051208496  | 30.0       |  
| stack_gated_cnn   | 0.9148936170212766   | 0.23397088050842285  | 30.0       |  
| rcnn              | 0.9042553191489362   | 0.19357776641845703  | 30.0       |  
| bert              | 0.851063829787234    | 0.743708610534668    | 30.0       |  
| bert_lstm         | 0.8936170212765957   | 1.2272799015045166   | 30.0       |  
| bert_cnn          | 0.8723404255319149   | 0.838465690612793    | 30.0       |  
| bert_mid_layer    | 0.8617021276595744   | 0.7447624206542969   | 30.0       |

```

