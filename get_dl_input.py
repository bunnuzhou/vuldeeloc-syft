## coding: utf-8
'''
该python文件用于将样本集划分为80%训练集和20%测试集，把原始语料数据和对应的词向量做替换，生成可以输入dl模型中的格式保存起来。
This python file is used to split database into 80% train set and 20% test set, tranfer the original code into vector, creating input file of deap learning model.
================================================================
原始文件：SySeVR/data_preprocess/get_dl_input.py
程序编写：Yoki
语言：Python 3.6
编写时间：2018.06.15

Original file: SySeVR/data_preprocess/get_dl_input.py
Original coder: Sophie
coder: Yoki
Language: Python 3.6
Date: 2018.06.15
================================================================
Log
-------------------------------------
* 2018.06.15  Yoki
modify the original python file to adapt python 3.6
注意：本python文件中使用gensim的3.4.0版本
* 2018.07.13  Yoki
adapt LLVM corpus files
'''
from __future__ import print_function
from gensim.models import Word2Vec
import numpy as np
import pickle
import os
import gc

VECTOR_DIM = 30  #token的向量维度
MAXLEN = 900   #切片的长度

def generate_corpus(model, sample):
    """generate corpus
    生成dl模型可以接受的输入
    
    This function is used to create input of deep learning model

    # Arguments
        #w2vModelPath: String, the path of word2vec or doc2vec model     //词向量模型或句向量模型保存的路径
        model : word2vec model
        samples: List, the samples                                      //样本列表
    
    # Return
        dl_corpus: the vectors of corpus                                //返回语料对应的向量
    """
    
    #model = Word2Vec.load(w2vModelPath)
    #dl_corpus = [[model[word] for word in sample]]
    dl_corpus = []
    for word in sample:
        if word in model:
            dl_corpus.append(model[word])
        else:
            dl_corpus.append([0]*VECTOR_DIM)

    return [dl_corpus]

#划分训练集和测试集
def get_dldata(filepath, dlTrainCorpusPath, dlTestCorpusPath, seed=2018, batch_size=16):
    """create deeplearning model dataset
    读取数据生成训练集和测试集
    
    This function is used to create train dataset and test dataset

    # Arguments
        filepath: String, path of all vectors           // 转好向量的数据集路径
        dlTrainCorpusPath: String, path of train set    // 训练集保存路径
        dlTestCorpusPath: String, path of test set      // 测试集保存路径
        seed: seed of random                            // 随机数种子
        batch_size: the size of mini-batch              // mini-batch的大小
    
    """
    #读取训练集测试集
    f = open("./record/dir_train.pkl",'rb')
    folders_train = pickle.load(f)
    print(folders_train)
    f.close()
    f = open("./record/dir_test.pkl",'rb')
    folders_test = pickle.load(f)
    print(folders_test)
    f.close()
    '''folders_test = ["CVE-2015-4504", "CVE-2013-7021", "CVE-2014-8544", "CVE-2014-7933", "CVE-2013-0845", "CVE-2013-0868", "CVE-2013-7019", "CVE-2013-7023", "CVE-2014-2097", "CVE-2014-7937", "CVE-2014-8544"]
    '''
    #x = os.listdir(filepath)
    #folders_test = [x for x in folders_test if not x in folders_train]
    #print(folders_test)
    #生成并保存训练集
    print("produce train dataset...")
    #由于数据量过大，每类分为N份
       
    N = 6
    num = list(range(N))
    for i in num:
        train_set = [[], [], [], [], [], []]
        for folder_train in folders_train[int(i*len(folders_train)/N) : int((i+1)*len(folders_train)/N)]:
            print(folder_train)
            if not folder_train in os.listdir(filepath):
                continue
            print("\rdzl"+str(folder_train), end='')
            for filename in os.listdir(os.path.join(filepath, folder_train)):
                f = open(filepath + folder_train + '/' + filename, 'rb')
                data = pickle.load(f)
                f.close()
                if len(data[0][0]) > MAXLEN:
                    #如果该切片超长被截断，抛去超长范围外的漏洞点
                    data[2] = [x for x in data[2] if x <= MAXLEN]
                data[0] = cutdata(data[0][0])
                if data[0] == None:
                    continue        
                for n in range(len(data)):
                    train_set[n].append(data[n])
                #记录testcase切片名
                train_set[-1].append(folder_train+"/"+filename)
        #print(train_set)
        #训练数据做随机打乱处理
        for x in train_set:
            np.random.seed(seed)
            np.random.shuffle(x)
        
        #保存训练集
        f_train = open(dlTrainCorpusPath + "train_" + str(i)+ "_0818.pkl", 'wb')
        pickle.dump(train_set, f_train)
        f_train.close()
        #print(train_set)
        del train_set #删除train_set
        gc.collect() #强制垃圾回收
    #return
    #生成并保存测试集

    print("\nproduce test dataset...")
    #由于数据量过大，每类分为N份
    N = 6
    num = list(range(N))
    for i in num:
        test_set = [[], [], [], [], [], []]
        for folder_test in folders_test[int(i*len(folders_test)/N) : int((i+1)*len(folders_test)/N)]:
            if not folder_test in os.listdir(filepath):
                continue
            print("\r"+str(folder_test), end='')
            for filename in os.listdir(os.path.join(filepath, folder_test)):
                f = open(filepath + folder_test + '/' + filename, 'rb')
                data = pickle.load(f)
                f.close()
                if len(data[0][0]) > MAXLEN:
                    #如果该切片超长被截断，抛去超长范围外的漏洞点
                    data[2] = [x for x in data[2] if x <= MAXLEN]
                data[0] = cutdata(data[0][0])
                if data[0] == None:
                    continue        
                for n in range(len(data)):
                    test_set[n].append(data[n])
                #记录testcase切片名
                test_set[-1].append(folder_test+"/"+filename)
            
        #保存测试集
        f_test = open(dlTestCorpusPath + "test_" + str(i)+ "_0124.pkl", 'wb')
        pickle.dump(test_set, f_test)
        f_test.close()
        #print(test_set)
        del test_set
        gc.collect()
    return

def cutdata(data, maxlen=MAXLEN, vector_dim=VECTOR_DIM):
    """cut data to maxlen
    对切片做截取
    
    This function is used to cut the slice or fill slice to maxlen

    # Arguments
        data: The slice
        maxlen: The max length to limit the slice
        vector_dim: the dim of vector
    
    """
    if maxlen:
        fill_0 = [0]*vector_dim
    
        #大于900的数据删除（才怪
        if len(data) > 900:
            pass
        #小于maxlen的数据补零；等于maxlen的不做处理
        if len(data) <=  maxlen:
            data = data + [fill_0] * (maxlen - len(data))
        else:
        #大于maxlen的数据做截取，直接截取maxlen长度
            data = data[:maxlen]
    return data

if __name__ == "__main__":
    
    CORPUSPATH = "./data/NVD/corpus_NVD/"
    VECTORPATH = "./data/vector_nvd/"
    W2VPATH = "./w2v_model/wordmodel_min_iter5.model"
    
    
    #将语料库转为向量
    print("turn the corpus into vectors...")
    model = Word2Vec.load(W2VPATH)
    #对于每一个程序id的文件夹
    for testcase in os.listdir(CORPUSPATH):
        print("\r" + testcase, end='')
        if testcase not in os.listdir(VECTORPATH):    #如果路径不存在则创建路径
            folder_path = os.path.join(VECTORPATH, testcase)
            os.mkdir(folder_path)
        #对于每一个程序id文件夹中的每一个文件
        for corpusfile in os.listdir(CORPUSPATH + testcase):
            #读取语料
            corpus_path = os.path.join(CORPUSPATH, testcase, corpusfile)
            f_corpus = open(corpus_path, 'rb')
            data = pickle.load(f_corpus)
            f_corpus.close()
            #转向量
            data.append(data[0])
            data[0] = generate_corpus(model, data[0])
            #存储
            vector_path = os.path.join(VECTORPATH, testcase, corpusfile)
            f_vector = open(vector_path, 'wb')
            pickle.dump(data, f_vector)
            f_vector.close()
    print("\nw2v over...")
    
    #生成训练集和测试集
    print("spliting the train set and test set...")
    dlTrainCorpusPath = "data/dl_input/nvd/train_NVD/"
    dlTestCorpusPath = "data/dl_input/nvd/test/"
    get_dldata(VECTORPATH, dlTrainCorpusPath, dlTestCorpusPath)
    
    print("\nsuccess!")
