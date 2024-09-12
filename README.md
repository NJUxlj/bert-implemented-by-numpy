# bert-projects

基于BERT模型的各种项目的集合

---

### numpy_bert.py

单纯使用numpy来手搓一个Bert模型

---


### numpy_ngram.py
单纯使用numpy来手搓一个ngram模型

---


### bert_lstm_pipeline
- 项目名称：基于Bert+LSTM+TextCNN的电商评论分类系统
- 项目要求：
  - 电商评论分类：好评/差评
  - 训练集/验证集划分
  - 数据分析：正负样本数，文本平均长度等
  - 实验对比3种以上模型结构的分类效果
  - 每种模型对比模型预测速度
  - 总结成表格输出
- 使用深度学习项目pipeline格式实现了多种模型【手搓Bert、TextRCNN、LSTM等】的多卡并行训练
- pipeline经典框架格式：【config.py, loader.py, model.py, evaluate.py, main.py】。
  实现解析config中的模型参数，并在loader中选择不同的文本分类模型, main中定义了统一的训练流程。
- 在model.py中，我们实现了Bert， LSTM, Bert+LSTM, TextRCNN、GRU、Gated-CNN 等文本分类模型， 并且，我们根据config文件中“model”字段的配置，自动选择加载某个模型进行训练。
- 同时，在运行时程序会自动遍历所有【model_type, lr, hidden_size, batch_size, pooling_type】的组合，并且log每一轮的配置+得分, 最后输出csv文件。


---


### bert-document-classification-reproduce
我复现了一篇Bert相关论文《用知识图嵌入丰富BERT进行文档分类》的代码，并对代码逻辑做出了改进。

#### 项目内容
- 本项目主要研究使用简短描述性文本（封面简介）和附加元数据对书籍进行分类。
- 作者演示了如何将文本表示（text-representation）与元数据(meta-data)和知识图嵌入(knowledge-graph-embeddings)相结合，并使用上述手段对作者信息进行编码。
- 评测集：2019 GermEval shared task on hierarchical text classification （即，共享任务数据集）
- 我们采用BERT进行文本分类，并在共享任务的上下文中提供额外的元数据来扩展模型，如作者、出版商、出版日期等。
- 注意，作者仅仅使用了书的简介来作为分类模型的输入。

#### 原文模型结构
- 新模型 = 原始Bert模型+双层MLP分类器+输出层softmax
- 输入数据分为两种：文本特征、非文本特征。
- Bert模型的输入格式 = 书籍标题+书籍简介，
- MLP分类器的输入：bert的输出+元数据特征+作者信息嵌入

#### 我做出的改进
1. 使用PCA和Forward-Selection降维技术进一步提炼 text-feature, extra featrue 这两块特征的组合。
2. 我提出要使用 BERT + TextRCNN(LSTM+TextCNN) + MLP + SoftMAX 来替代原来的模型结构
3. 原因：
  - 我们分析， 在一些文本分类任务重， 双向编码可能并不是必要的，可能会产生对目标token的过度预测，因此有必要实验原始的LSTM， 同时保留语序信息。
  - 另外， TextCNN在pooling时也能有效保留语序信息。



## 运行项目
```shell
conda create --name myenv python=3.10
conda activate myenv

# cd 到项目根目录下
pip install -r requirements.txt  
```
