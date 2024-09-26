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



