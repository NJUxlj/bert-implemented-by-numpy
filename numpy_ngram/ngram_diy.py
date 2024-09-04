import numpy as np
import math
from collections import defaultdict

import torch
import torch.nn as nn

from typing  import Dict



class NgramLanguageModel(nn.Module):
    
    
    def __init__(self, corpus = None, n=3):
        self.n = n 
        self.sep = '-'
        self.sos = '<sos>' # start token of a sentence
        self.eos = '<eos>'
        self.unk_prob = 1e-5  # give Unregistered words a probability
        self.fix_backoff_prob = 0.4 
        
        self.ngram_count_dict:Dict[int][str] = dict((x+1,defaultdict(int)) for x in range(n) )
        self.ngram_count_prob_dict:Dict[int][str] = dict((x+1, defaultdict(int)) for x in range(n))

        
        self.ngram_count(corpus)
        self.calc_ngram_probs()

        
            
    
    

    def sentence_segments(self, sentence):
        return sentence.split()
    
    
    def ngram_count(self, corpus):
        
        '''
         统计 每个窗口长度对应的每个 ngram的数量
        '''
        for sentence in corpus:
            word_lists = self.sentence_segments(sentence)
            word_lists = [self.sos] + word_lists + [self.eos]

            for window_size in range(1, self.n+1):
                for index, word in enumerate(word_lists):
                    # 遍历每个词，以每个词为开头，去一个ngram
                    if len(word_lists[index: index+window_size])!= window_size:
                        continue
                    
                    # 统计当前ngram的次数
                    
                    ngram = self.sep.join(word_lists[index:index+window_size])
                    self.ngram_count_dict[window_size][ngram]+=1
            
        # 统计完毕, 返回一阶grams的总数量 
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        return
    
            
            
    def calc_ngram_probs(self):
        for window_size in range(1, self.n+1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngram_splits = ngram.split(self.sep)
                    ngram_prefix = self.sep.join(ngram_splits[:-1])
                    ngram_prefix_count = self.ngram_count_dict[window_size-1][ngram_prefix]
                else :
                    ngram_prefix_count = self.ngram_count_dict[0]
                
            self.ngram_count_prob_dict[window_size][ngram] = count / ngram_prefix_count
        
        return
            
    
    
    
    def get_ngram_probs(self, ngram:str):
        n = len(ngram.split(self.sep))
        
        if ngram in self.ngram_count_prob_dict[n]:

            # 直接取出概率
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1 :  
            return self.unk_prob
        else:
            # 高于一阶，回退
            ngram = self.sep.join(ngram.split(self.sep)[1:])
            
            return self.fix_backoff_prob * self.get_ngram_probs(ngram)
        
  
        
    def calc_sentence_ppl(self, sentence):
        '''
        
         使用回退法 来预测成句概率
        '''
        word_list = self.sentence_segments(sentence)

        word_list = [self.sos] +word_list +[self.eos]

        sentence_prob = 0
        
        for index, word in enumerate(word_list):
            # 
            ngram = self.sep.join(word_list[max(0, index -self.n +1):index+1])
            prob = self.get_ngram_probs(ngram)
            
            sentence_prob += math.log(prob)
        
        return 2**(sentence_prob * (-1/len(word_list)))
            
    
    
        
        






if __name__ == '__main__':
    corpus = open(r"sample.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    print("total words number：", lm.ngram_count_dict[0])
    print(lm.ngram_count_prob_dict)
    print(lm.calc_sentence_ppl("今天你还好吗？"))
    

    
