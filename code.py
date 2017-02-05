# -*- coding: utf-8 -*-
"""
@author: moliu
"""
#引用库
import jieba
import pandas as pd
import word2vec

#读入数据
data = pd.read_csv('F://DataScience//xinlangweibo//TextMining//data.csv',names=['comment','label'],encoding='UTF-8')

#分词
file = open('comments.txt','w')
for i in data['comment']:    
    seg_list = jieba.cut(i, cut_all=False) #精确模式
    k = u' '.join(seg_list).encode('utf-8')
    file.write(k+"\n")
file.close()

# 构建词向量
model = word2vec.word2vec('comments.txt','corpusWord2Vec.bin', size=400,verbose=True)
model = word2vec.load('corpusWord2Vec.bin')
rawWordVec = model.vectors  #词向量
y=model.vocab    #词表中的词
print (model.vectors)
#显示空间距离相近的词
indexes = model.cosine(u'家长')  #试验了孩子、医生、家长
for index in indexes[0]:
    print (model.vocab[index])


#降维
from sklearn.decomposition import PCA
# reduce the dimension of word vector
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

#将分词之后的文档导入
#from pandas import read_table
#df=read_table('comments.txt',names=['comment'])








