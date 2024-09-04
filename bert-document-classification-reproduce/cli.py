
"""

Run experiments for paper results
        - for each experiment: sub-task A + B
- save model weights to disk
- save val results to disk

/experiments/
        bert_extras/
                weights/
                config/
                validation/
                stats/


export BERT_MODELS_DIR="/home/mostendorff/datasets/BERT_pre_trained_models/pytorch"

python cli.py run_on_val <name> $GPU_ID $EXTRAS_DIR $TRAIN_DF_PATH $VAL_DF_PATH $OUTPUT_DIR --epochs 5

    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-a__bert-german_manual_author-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output


python experiments.py run task-a__bert-german_manual_no-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output
    
python experiments.py run task-a__bert-german_manual_no-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output


python experiments.py run task-a__bert-german_text-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-a__author-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-b__author-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-b__bert-german_full \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output \
    --epochs 5

----

python experiments.py final task-a__bert-german_full \
    4 \
    data/extras \
    germeval_fulltrain_df_meta.pickle \
    germeval_test_df_meta.pickle \
    experiments_output \
    --epochs 1

python experiments.py final task-b__bert-german_full     3     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5


python experiments.py final task-a__bert-german_text-only     2     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5


python experiments.py final task-b__bert-german_full     2     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5

"""





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
# author dim: author-only-model 的hidden_size
from config import AUTHOR_DIM, LEARNING_RATE, TASK_A_LABELS_COUNT, TASK_B_LABELS_COUNT, most_popular_label
from data_utils import get_best_thresholds, nn_output_to_submission
from experiment import Experiment

# 导入MLP双层分类器
from models import LinearMultiClassifier

logging.basicConfig(level=logging.INFO)



# 定义实验 sub-task-A or B, bert+author embedding, meta data feature? 定义了用什么{模型+输入数据}的配置

experiments = {

  # ============== A ========================
    
  # full 代表 meta-data(author_gender+manual) + author_embedding(author_vec) 全部都输入
  
    'task-a__bert-german_full': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True
    ),

  # 2 代表两层MLP分类器
    'task-a__bert-german_full_2': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True, mlp_dim=500,
    ),

  # no-embedding 表示不要author-embedding
    'task-a__bert-german_manual_no-embedding': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=False
    ),
  
  # 表示不要meta-data
    'task-a__bert-german_no-manual_embedding': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=True
    ),

  # 除了标题和简介，其他都不要
    'task-a__bert-german_text-only': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),
    # author only model: 作者专门为author_embedding向量创建的分类器
    'task-a__author-only': Experiment(
        'a', '-', with_text=False, with_author_gender=False, with_manual=False, with_author_vec=True,
        classifier_model=LinearMultiClassifier(
            labels_count=TASK_A_LABELS_COUNT,
            extras_dim=AUTHOR_DIM,
        )
    ),
  
    # bert-base-multilingual-cased
    'task-a__bert-multilingual_text-only': Experiment(
        'a', 'bert-base-multilingual-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),






  # =================== B =====================

'task-b__bert-german_full': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True
    ),
    'task-b__bert-german_manual_no-embedding': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=False
    ),
    'task-b__bert-german_no-manual_embedding': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=True
    ),
    'task-b__bert-german_text-only': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),
    # author only
    'task-b__author-only': Experiment(
        'b', '-', with_text=False, with_author_gender=False, with_manual=False, with_author_vec=True,
        classifier_model=LinearMultiClassifier(
            labels_count=TASK_B_LABELS_COUNT,
            extras_dim=AUTHOR_DIM,
        )
    ),
    # bert-base-multilingual-cased
    'task-b__bert-multilingual_text-only': Experiment(
        'b', 'bert-base-multilingual-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),

    ######

    # switch does not work (??)
    'task-a__bert-german_full-switch': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True,
        author_vec_switch=True,
    ),


}
