import json
import os
import pickle
import numpy as np

import fire
import torch
import logging

from torch.optim import Adam
from sklearn.metrics import classification_report

# 从配置文件导入实验的超参数
from config import AUTHOR_DIM, LEARNING_RATE, TASK_A_LABELS_COUNT, TASK_B_LABELS_COUNT, most_popular_label
from data_utils import get_best_thresholds, nn_output_to_submission
from experiment import Experiment

# 导入MLP双层分类器
from models import LinearMultiClassifier

logging.basicConfig(level=logging.INFO)



# 定义实验 sub-task-A or B, bert+author embedding, meta data feature? 定义了用什么{模型+输入数据}的配置

experiments = {

  # ============== A ========================















  # =================== B =====================





  



}
