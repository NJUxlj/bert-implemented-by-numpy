## 复现并改进一篇论文：Enriching BERT with Knowledge Graph Embeddings for Document Classification

### 原项目地址
[pytorch-bert-document-classification](https://github.com/malteos/pytorch-bert-document-classification)

### 论文地址：
[Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402)


### 项目内容
- 本项目主要研究使用简短描述性文本（封面简介）和附加元数据对书籍进行分类。

- 作者演示了如何将文本表示（text-representation）与元数据(meta-data)和知识图嵌入(knowledge-graph-embeddings)相结合，并使用上述手段对作者信息进行编码。
- 评测集：2019 GermEval shared task on hierarchical text classification （即，共享任务数据集）
- 我们采用BERT进行文本分类，并在共享任务的上下文中提供额外的元数据来扩展模型，如作者、出版商、出版日期等。

- 注意，作者仅仅使用了书的简介来作为分类模型的输入。


### 项目主要贡献
- 使用最先进的文本处理方法纳入了额外的（元数据）数据。

### 原文模型结构
1. 新模型 = 原始Bert模型+双层MLP分类器+输出层softmax
2. 输入数据分为两种：文本特征、非文本特征。
3. 文本特征：标题+简介，不超过300 tokens
4. 非文本特征包含：{元数据特征：10维向量（其中有2维代表性别）、作者信息嵌入：200维向量}
5. Bert模型的输入格式 = 书籍标题+书籍简介，
6. MLP分类器的输入：bert的输出+元数据特征+作者信息嵌入
![image](https://github.com/user-attachments/assets/9bc7a6a2-bf28-49c6-97be-5bbaf3d401e9)


### 训练过程


### 原实验结果



### 我做出了如下改进
#### 改进1
- 原始项目使用了德语写的书籍的简介作为语料，并且最终预训练了一个德语模型。具体来说，该模型是在德国维基百科、新闻文章和法院判决书上从头开始训练的。
- 我将训练语料改为中文，并且想输入格式与国内习惯相匹配



### 实验结果
