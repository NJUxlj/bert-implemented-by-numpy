## 复现并改进一篇论文：Enriching BERT with Knowledge Graph Embeddings for Document Classification

### 原项目地址
[pytorch-bert-document-classification](https://github.com/malteos/pytorch-bert-document-classification)

### 论文地址：
[Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402)


### 项目内容
- 本项目主要研究使用简短描述性文本（封面简介）和附加元数据对书籍进行分类。

- 作者演示了如何将文本表示（text-representation）与元数据(meta-data)和知识图嵌入(knowledge-graph-embeddings)相结合，并使用上述手段对作者信息进行编码。

- 我们采用BERT进行文本分类，并在共享任务（评测集：2019 GermEval shared task on hierarchical text classification）的上下文中提供额外的元数据来扩展模型，如作者、出版商、出版日期等。

- 注意，作者仅仅使用了书的简介来作为分类模型的输入。
### 项目主要贡献
- 使用最先进的文本处理方法纳入了额外的（元数据）数据。

### 原文模型结构
![image](https://github.com/user-attachments/assets/9bc7a6a2-bf28-49c6-97be-5bbaf3d401e9)


### 项目重点


### 我做出了如下改进




### 实验结果
