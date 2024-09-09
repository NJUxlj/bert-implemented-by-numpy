import numpy as np
from collections import defaultdict
import json

import jieba

jieba.initialize()

class BayesApproach:
    def __init__(self, data_path):
        self.p_class = defaultdict(int)
        self.word_class_prob = defaultdict(dict)
        self.load(data_path)
        
        
    def load(self, path:str):
        self.class_name_to_word_freq = defaultdict(dict)
        self.all_words  = set()
        with open(path, 'r', encoding = 'utf8') as f:
            for line in f:
                line = json.loads(line)
                title = line['title']
                class_name = line['tag']
                words = jieba.lcut(title)
                self.all_words = self.all_words.union(set(words))
                self.p_class[class_name]+=1
                
                words_freq = self.class_name_to_word_freq[class_name]
                # 记录每个类别下的词频
                for word in words:
                    if word not in words_freq:
                        words_freq[word] = 1
                    else:
                        words_freq[word]+=1
        self.freq_to_prob()
        return
    
    def freq_to_prob(self):
        
        # 先计算每个类别的概率
        total_sample_count = sum(self.p_class.values())
        self.p_class = dict([name, freq/total_sample_count] for name, freq in self.p_class.items())

        # 计算词属于类的概率
        self.word_class_prob = defaultdict(dict)
        for class_name, word_freq in self.class_name_to_word_freq.items():
            total_words_in_class = sum(count for count in word_freq.values())
            for word in word_freq:
                # 加一平滑, 对象：词表中有，但是新闻标题中没出现的那些词
                prob = (word_freq[word]+1)/(total_words_in_class+len(self.all_words))
                self.word_class_prob[class_name][word] = prob
            self.word_class_prob[class_name]["<unk>"] = 1/(total_words_in_class+len(self.all_words))
        return 
    
    
    #P(w1|x1) * P(w2|x1)...P(wn|x1)
    def get_word_given_class_probs_product(self, words, class_name):
        result = 1
        for word in words:
            unk_prob = self.word_class_prob[class_name]["<unk>"]
            result *= self.word_class_prob[class_name].get(word, unk_prob)
        return result
    
    #计算P(w1, w2..wn|x1) * P(x1)
    def get_sentence_given_class_prob(self, words, class_name):
        p_x1 = self.p_class[class_name]
        
        return self.get_word_given_class_probs_product(words, class_name) * p_x1
    
    
    def classify(self, sentence) -> str:
        '''
           做文本分类
        '''
        words = jieba.lcut(sentence)
        results = []
        
        for name, prob in self.p_class.items():
            sentence_prob = self.get_sentence_given_class_prob(words, name)
            results.append((name, sentence_prob))
        
        results:list = sorted(results, key=lambda x:x[1], reverse=True)
        
        print("result = ", results)
        
        # 计算公共分母 P(w1, w2, w3...wn) = P(w1,w2..Wn|x1)*P(x1) + P(w1,w2..Wn|x2)*P(x2) ... P(w1,w2..Wn|xn)*P(xn)
        # P(x1 | w1, w2, w3..wn) = P(x1) * P(w1,w2..Wn|x1) / P(w1, w2, w3..wn)
        common_divisor = 0
        for name, prob in results:
            common_divisor += prob
            
        print("common_divisor = ", common_divisor)
        results = map(lambda x: (x[0], x[1]/common_divisor), results)
        
    
        
        # 打印结果
        for class_name, prob in results:
            print("属于类别[%s]的概率为%f" % (class_name, prob))
        return results



if __name__ == "__main__":
    data_path = "data/tag_news.json"
    
    bayes = BayesApproach(data_path) 
    query = "中国三款导弹可发射多弹头 美无法防御很急躁"
    bayes.classify(query)
