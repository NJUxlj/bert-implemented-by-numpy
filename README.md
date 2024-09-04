# bert-projects
基于BERT模型的各种项目的集合

### numpy_bert.py
单纯使用numpy来手搓一个Bert模型


### numpy_ngram.py
单纯使用numpy来手搓一个ngram模型



### bert_lstm_pipeline
- 使用深度学习项目pipeline格式实现了多种模型【手搓Bert、TextRCNN、LSTM等】的多卡并行训练
- pipeline经典框架格式：【config.py, loader.py, model.py, evaluate.py, main.py】。
  实现解析config中的模型参数，并在loader中选择不同的文本分类模型, main中定义了统一的训练流程。
  在model.py中，我们实现了Bert， LSTM, Bert+LSTM, TextRCNN、GRU、Gated-CNN 等文本分类模型， 并且，我们根据config文件中“model”字段的配置，自动选择加载某个模型进行训练。同时，在运行时程序会自动遍历所有【model_type, lr, hidden_size, batch_size, pooling_type】的组合，并且log每一轮的配置+得分, 最后输出csv文件。




### bert-document-classification-reproduce
我复现了一篇Bert相关论文《用知识图嵌入丰富BERT进行文档分类》的代码，并对代码逻辑做出了改进。
