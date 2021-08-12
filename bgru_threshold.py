## coding: utf-8
'''
该python文件用于SySeVR+训练BGRU模型
This python file is used to train four class focus data in bgru model
================================================================
原始文件：SySeVR/BLSTM/bgru_3+1_analyze.py
代码修改：Yoki
语言：Python 3.6
编写时间：2018.06.04

Original file: SySeVR/BLSTM/bgru_3+1_analyze.py
programmmer: Yoki
Language: Python 3.6
Date: 2018.06.04
================================================================
Log
-------------------------------------
* 2018.06.04  Yoki
version 0.1 to train BGRU rely on Pooling
* 2018.06.05  Yoki
version 0.2 to run in python3.6 and backend is tensorflow
模型顶层施加池化机制
* 2018.06.11  Yoki
version 0.3 to modify model to functional model
* 2018.07.14  Yoki
adapt LLVM
* 2018.08.04  Yoki
modify the way of train and test based on line
'''

from __future__ import absolute_import
from __future__ import print_function
from keras import metrics
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta#optimizers是优化器，是调整节点权重的方法（sdg是随机梯度下降算法）
from keras.models import Sequential, load_model, Model#model层包括sequential模型（多个网络层的线性堆叠）和泛型模型
from keras.layers import Input, Multiply
from keras.layers.core import Masking, Dense, Dropout, Activation, Lambda, Reshape#core是常用层（dropout层用于防止过拟合）
from keras.layers.recurrent import GRU, LSTM#recurrent是递归层（包含LSTM,GRU等）
from keras.layers.pooling import GlobalAveragePooling1D#pooling是池化层
from keras.engine.topology import Layer, InputSpec
from preprocess_dl_Input_version4 import *
from keras.layers.wrappers import Bidirectional, TimeDistributed#wrappers是包装器，TimeDistributed可以把一个层应用到输入的每一个时间步上，Bidirectional是双向RNN包装器
from collections import Counter
import tensorflow as tf
import keras.backend as K
import numpy as np
import pickle
import random
import time
import math
import os


RANDOMSEED = 2018  # for reproducibility
#GPU使用
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class NonMasking(Layer):
    """Non Masking Layer
    自定义的消除蒙版层，以应对池化不支持蒙版
    """
    
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)
    
    def build(self, input_shape):
        input_shape = input_shape
    
    def compute_mask(self, input, input_mask=None):
        #不传输蒙版
        return None
    
    def call(self, x, mask=None):
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
class KMaxPooling(Layer):
    """K-max pooling
    k-max池化
    
    k-max pooling layer that extracts the k-highest activations from a sequence and calculate average
    base on Tensorflow backend.
    """
    def __init__(self, k=1, **kwargs):
        super(KMaxPooling,self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim = 3)
        self.k = k
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[1])
        
    def call(self, inputs):
        #选出k-top个值
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]
        #对矩阵后两维取转置
        shifted_output = tf.transpose(top_k, [0, 2, 1])
        #返回转置后的top_k
        return shifted_output

def build_model(maxlen, vector_dim, dropout=0.4):
    """build model
    建立模型
    
    Build the model according to the arguments.
    按照参数建立模型并返回该模型
    
    # Arguments                                         // 参数列表
        maxlen: The max length of a sample              // 输入模型的单个样本的最大长度
        vector_dim: The size of token's dim             // 输入模型的单个样本中单个token的向量维度
        dropout : the rate of dropout                   // 模型dropout的比例（在模型训练时随机让某些隐含层节点的权重不工作的比例）
        winlen : the length of average pooling windows  // 平均池化的窗口大小
    
    # Returns
        model : model just build
    """
    
    print('Build model...')
    
    #对于漏洞样本输入，产生经过BGRU模型的序列
    inputs = Input(shape=(maxlen, vector_dim))
    mask_1 = Masking(mask_value=0.0, name='mask_1')(inputs)
    bgru_1 = Bidirectional(GRU(units=512, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True), name='bgru_1')(mask_1)
    dropout_1 = Dropout(dropout, name='dropout_1')(bgru_1)
    bgru_2 = Bidirectional(GRU(units=512, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True), name='bgru_2')(dropout_1)
    dropout_2 = Dropout(dropout, name='dropout_2')(bgru_2)
    dense_1 = TimeDistributed(Dense(1), name='dense1')(dropout_2)
    activation_1 = Activation('sigmoid', name='activation_1')(dense_1)
    unmask_1 = NonMasking()(activation_1)
    
    #对于漏洞位置的输入（漏洞位置矩阵）进行乘积操作并选出k-max个最高值后取平均
    vulner_mask_input = Input(shape=(maxlen, maxlen), name='vulner_mask_input')  #输入漏洞位置矩阵
    multiply_1 = Multiply(name='multiply_1')([vulner_mask_input, unmask_1])  #漏洞位置矩阵与激活层输出向量做乘积
    reshape_1 = Reshape((1, maxlen ** 2))(multiply_1)  
    k_max_1 = KMaxPooling(k=1, name='k_max_1')(reshape_1)  #过滤掉所有不存在漏洞的token，在剩下的值中取最大的前K个值
    average_1 = GlobalAveragePooling1D(name='average_1')(k_max_1) #对这个K个值取平均，即得到一个值，该值即表示该切片样本中可能存在漏洞的分数
    
    #模型编译
    print("begin compile")
    model = Model(inputs=[inputs, vulner_mask_input], outputs=average_1)
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['TP_count', 'FP_count', 'FN_count', 'precision','recall', 'fbeta_score'])
    #打印模型结构
    model.summary()
 
    return model


def main(traindataSet_path, testdataSet_path, weightpath, resultpath, batch_size, maxlen, vector_dim, **kw):  #**kw代表dropout的大小
    """Train model and test model
    训练模型和测试模型
    
    Build the model and preprocess dataset for training, then train the model
    using the train dataset and save them to the weight path, test the test
    dataset and save the test result into result path.
    建立模型并预处理数据，将训练集输入模型中训练，保存模型至文件，对测试数据集进行测试，保存
    测试结果于指定文件。
    
    # Arguments                                         // 参数列表
        traindataSet_path: The path of train dataset    // 训练数据集路径
        testdataSet_path: The path of test dataset      // 测试数据集路径
        weightpath: The path to save the model          // 保存模型路径
        resultpath: The path to save the test result    // 测试结果保存路径
        batch_size: The size of minibatch               // 输入模型的mini-batch的大小
        maxlen: The max length of a sample              // 输入模型的单个样本的最大长度
        vector_dim: The size of token's dim             // 输入模型的单个样本中单个token的向量维度
        **kw : the dict arguments for model, like layers and so on  // 模型的各项参数例如层数、dropout、池化窗口的大小等
    
    # Returns
        None
    """
    
    #建立模型
    model = build_model(maxlen, vector_dim, **kw)
    
    #如果需要加载模型
    #model.load_weights(weightpath)
    
    #训练阶段开始
    #加载训练数据
    dataset = []
    labels = []
    linetokens = []
    vpointers = []
    funcs = []
    print("Loading data...")
    #逐个读取训练路径下各个训练数据文件并合并入同一结构
    for filename in os.listdir(traindataSet_path):
        if(filename.endswith(".pkl") is False):
            continue
        print(filename)
        f = open(os.path.join(traindataSet_path,filename),"rb+")
        #分别是样本向量、句子拆分点、漏洞点、goodbad函数表、句子语料、testcase切片名
        print(f)
        dataset_file, linetokens_file, vpointers_file, func_file, corpus_file, testcase_file = pickle.load(f,encoding = 'iso-8859-1')
        f.close()
        dataset += dataset_file
        linetokens += linetokens_file
        vpointers += vpointers_file
        funcs += func_file
    print(len(dataset))
    
    #按照漏洞点和所属函数生成0/1标签
    for vp in range(len(vpointers)):
        if vpointers[vp] != []:
            label = 1
            for func in funcs[vp]:
                #如果是good函数中提的切片实际上无漏洞
                if "good" in func:
                    label = 0
                    vpointers[vp] = []
                    break
        else:
            label = 0
        labels.append(label)

    #生成训练生成器
    train_generator = generator_of_data(dataset, labels, linetokens, vpointers, batch_size, maxlen, vector_dim)
    all_train_samples = len(dataset)
    steps_epoch = int(all_train_samples / batch_size)
    
    #开始输入模型中训练，迭代4次
    print("Train...")
    model.fit_generator(train_generator, steps_per_epoch=steps_epoch, epochs=4)

    #保存模型
    model.save_weights(weightpath)
    
    #如果需要加载模型
    #model.load_weights(weightpath)
    
    #测试阶段开始
    dataset = []
    linetokens = []
    vpointers = []
    funcs = []
    testcase = []
    print("Test...")
    for filename in os.listdir(testdataSet_path):
        if(filename.endswith(".pkl") is False):
            continue
        print(filename)
        f = open(os.path.join(testdataSet_path,filename),"rb")
        #分别是样本向量、句子拆分点、漏洞点、goodbad函数表、句子语料、testcase切片名
        dataset_file, linetokens_file, vpointers_file, funcs_file, corpus_file, testcase_file = pickle.load(f,encoding = 'iso-8859-1')
        f.close()
        dataset += dataset_file
        linetokens += linetokens_file
        vpointers += vpointers_file
        funcs += funcs_file
        testcase +=testcase_file
    print(len(dataset),len(testcase))
    
        #按照漏洞点和所属函数生成0/1标签
    labels = []
    for vp in range(len(vpointers)):
        if vpointers[vp] != []:
            label = 1
            for func in funcs[vp]:
                if "good" in func:
                    label = 0
                    break
        else:
            label = 0   #无漏洞的切片的漏洞点标签为空
        labels.append(label)
        
        #测试模型
    batch_size = 64
    test_generator = generator_of_data(dataset, labels, linetokens, vpointers, batch_size, maxlen, vector_dim)
    all_test_samples = len(dataset)
    steps_epoch = int(all_test_samples / batch_size)
    #构建测试函数
    get_bgru_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[7].output]) #一层BGRU时读取第5层，两层BGRU时读取第7层，三层BGRU时读取第9层
    TP, TN, FP, FN = 0, 0, 0, 0
    TP_l, TN_l, FP_l, FN_l = 0, 0, 0, 0
    TP_index = []
    results = {}
    dict_testcase2func = {}
    start = time.time()
    #对于每一个样本做测试
	
    for i in range(steps_epoch):
        print("\r", i, "/", steps_epoch, end="")
        #测试输入
        test_input = next(test_generator)
        #深度学习模型的序列输出
        layer_output = get_bgru_output([test_input[0][0],0])
		
        #测试结果
        for j in range(batch_size):
            index = i*batch_size + j
            #print('\n',testcase[index])
            result = sample_threshold_windows(layer_output[0][j], linetokens[index], {'threshold_value':0.5, 'k':1})#result是漏洞行的行号
        
            #输出统计
            if result:
                y_pred = 1
            else:
                y_pred = 0
                        
        #漏洞分类的指标 & 漏洞分类的函数级指标 & 保存每个测试样本的序列输出
            if y_pred == 0 and labels[index] == 0:
                TN += 1
                TN_l += 1
                with open("BGRU/result_analyze_0.4_k1/TN/"+str(index)+".pkl","wb") as f:
                    pickle.dump(list([layer_output[0][j]]),f)
                with open("BGRU/result_analyze_0.4_k1/TN_testcase_id_0514.txt",'a+') as ftn:
                    ftn.write(testcase[index]+'\n')
            if y_pred == 0 and labels[index] == 1:
                FN += 1
                FN_l += 1
                with open("BGRU/result_analyze_0.4_k1/FN/"+str(index)+".pkl","wb") as f:
                    pickle.dump(list([layer_output[0][j]]),f)
                with open("BGRU/result_analyze_0.4_k1/FN_testcase_id_0514.txt",'a+') as ffn:
                    ffn.write(testcase[index]+'\n')
					
            if y_pred == 1 and labels[index] == 0:
                FP += 1
                FP_l += 1
                if not testcase[index].split("/")[0] in dict_testcase2func.keys():
                    dict_testcase2func[testcase[index].split("/")[0]]={}
                for func in funcs[index]:
                    if func in dict_testcase2func[testcase[index].split("/")[0]].keys():
                        dict_testcase2func[testcase[index].split("/")[0]][func].append("FP")
                    else:
                        dict_testcase2func[testcase[index].split("/")[0]][func] = ["FP"]
                with open("BGRU/result_analyze_0.4_k1/FP/"+str(index)+".pkl","wb") as f:
                    pickle.dump(list([layer_output[0][j]]),f)
                with open("BGRU/result_analyze_0.4_k1/FP_testcase_id_0514.txt",'a+') as ffp:
                    ffp.write(testcase[index]+'\n')					
                    
            if y_pred == 1 and labels[index] == 1:
                TP += 1
                TP_index.append(index)
                flag_l = False
            #基于切片中的漏洞行位置判断是否正确
                for pred in result:
                    if linetokens[index][pred] in vpointers[index]:
                        flag_l = True
                        break
                if flag_l:
                    TP_l += 1
                else:
                    FN_l += 1
                if not testcase[index].split("/")[0] in dict_testcase2func.keys():
                    dict_testcase2func[testcase[index].split("/")[0]]={}
                for func in funcs[index]:
                    if func in dict_testcase2func[testcase[index].split("/")[0]].keys():
                        dict_testcase2func[testcase[index].split("/")[0]][func].append("TP")
                    else:
                        dict_testcase2func[testcase[index].split("/")[0]][func] = ["TP"]
                with open("BGRU/result_analyze_0.4_k1/TP/"+str(index)+".pkl","wb") as f:
                    pickle.dump(list([layer_output[0][j]]),f)
                with open("BGRU/result_analyze_0.4_k1/TP_testcase_id_0514.txt",'a+') as ftp:
                    ftp.write(testcase[index]+'\n')
         
            results[testcase[index]] = result
    end = time.time()
    print(end - start)
        
        #保存预测的漏洞行行号
    with open(resultpath+"_result.pkl", 'wb') as f:
        pickle.dump(results, f)
        
        #保存testcase到函数的预测映射
    with open(resultpath+"_dict_testcase2func.pkl", 'wb') as f:
        pickle.dump(dict_testcase2func, f)
        
        #保存TP的结果
    with open("TP_index_0.4_k2_BGRU.pkl", 'wb') as f:
        pickle.dump(TP_index, f)
        
        #记录实验结果
    with open(resultpath, 'a') as fwrite:
            #实验基本信息
        fwrite.write('test_samples_num: '+ str(len(dataset)) + '\n')
        fwrite.write('train_dataset: ' + str(traindataSetPath) + '\n')
        fwrite.write('test_dataset: ' + str(filename) + '\n')
        fwrite.write('model: ' + weightPath + '\n')
        #漏洞分类指标
        fwrite.write('TP:' + str(TP) + ' FP:' + str(FP) + ' FN:' + str(FN) + ' TN:' + str(TN) + '\n')
        FPR = FP / (FP + TN)
        fwrite.write('FPR: ' + str(FPR) + '\n')
        FNR = FN / (TP + FN)
        fwrite.write('FNR: ' + str(FNR) + '\n')
        accuracy = (TP + TN) / (len(dataset))
        fwrite.write('accuracy: ' + str(accuracy) + '\n')
        precision = TP / (TP + FP)
        fwrite.write('precision: ' + str(precision) + '\n')
        recall = TP / (TP + FN)
        fwrite.write('recall: ' + str(recall) + '\n')
        f_score = (2 * precision * recall) / (precision + recall)
        fwrite.write('fbeta_score: ' + str(f_score) + '\n')
        #漏洞定位指标
        fwrite.write('TP_l:' + str(TP_l) + ' FP_l:' + str(FP_l) + ' FN_l:' + str(FN_l) + ' TN:' + str(TN_l) + '\n')
        FPR_l = FP_l / (FP_l + TN_l)
        fwrite.write('FPR_location: ' + str(FPR_l) + '\n')
        FNR_l = FN_l / (TP_l + FN_l)
        fwrite.write('FNR_location: ' + str(FNR_l) + '\n')
        accuracy_l = (TP_l + TN_l) / (TP_l + FP_l + FN_l + TN_l)
        fwrite.write('accuracy_location: ' + str(accuracy_l) + '\n')
        precision_l = TP_l / (TP_l + FP_l)
        fwrite.write('precision_location: ' + str(precision_l) + '\n')
        recall_l = TP_l / (TP_l + FN_l)
        fwrite.write('recall_location: ' + str(recall_l) + '\n')
        f_score_l = (2 * precision_l * recall_l) / (precision_l + recall_l)
        fwrite.write('fbeta_score_location: ' + str(f_score_l) + '\n')
        fwrite.write('--------------------\n')
        
    print("\nf1: ",f_score)
    
    
def sample_threshold_windows(value_sequence, linetokens, argv):
    """read the output of RNN deaplearning model output sequence and return the classify result
    
    Input the output of RNN model's top activation(sigmoid) layer,
    read them by time step, and judge the value according to the threshold
    value, count the average that value bigger than threshold, and return the
    classify result 1 or 0.
    根据模型的顶层输出窗口读取，窗口内平均值超过阈值则认定为有漏洞
    
    # Arguments
        value_sequence: The output sequence from RNN model
        linetokens: The index of first token each line      // 句首token索引值
        argv: The dict of argument, contain 'k' 'threshold_value'
    
    # Returns
        linenum: The line predict to have vulnerability's line num      // 报为有漏洞的切片行号
    """
        
    #检查参数是否有效
    if "threshold_value" in argv: #argv是参数列表，包括阈值和K值
        threshold_value = argv["threshold_value"]
    else:
        print("Error:Bad threshold value!")
        return -1
    if "k" in argv:
        k = argv["k"]
    else:
        k = 3
        
    #删除尾部重复数值
    value_sequence = list(value_sequence)
    #print('\n',len(value_sequence))
    vs = len(value_sequence)-1
    while value_sequence[vs] == value_sequence[-1]:
        vs -= 1
    value_sequence = value_sequence[:vs+2]
        
    #标注序列的终止位置为序列长度
    for tokenindex in range(len(linetokens)):
        if len(value_sequence) <= linetokens[tokenindex]:
            linetokens = linetokens[:tokenindex]
            linetokens.append(len(value_sequence))
            break
    if len(value_sequence) > linetokens[-1]:
        linetokens.append(len(value_sequence))
    
    #扫描每一行，输出序列窗口内k-max的平均值大于阈值则认定为有漏洞(对每一行取前K个最大值的平均值，平均值大于阈值就是有漏洞，取该漏洞行行号作为输出结果)
    linenum = []
    for i in range(len(linetokens)-1):
        left = linetokens[i]
        right = linetokens[i+1]
        if left == right:
            right += 1
        window = value_sequence[left: right]
        window = [x[0] for x in window]
        window.sort(reverse = True)
        k_max = window[:k]
        if sum(k_max)/ len(k_max) > threshold_value:
            #如果该行被认定为有漏洞
            linenum.append(i)
    
    return linenum
    
'''def testrealdata(realtestpath, weightpath, batch_size, maxlen, vector_dim, dropout):
    """This function is used to judge the real data(data without label) is vulnerability or not
    对未知真实数据（数据不带有标签）进行判别

    # Arguments
    realdataSet_path: String type, the data path to load real data set
    weight_path: String type, the data path of model
    batch_size: Int type, the mini-batch size
    maxlen: Int type, the max length of data
    vector_dim: Int type, the number of data vector's dim
    dropout: Float type. the value of dropout
    
    """
    #加载模型
    model = build_model(maxlen, vector_dim, dropout)
    model.load_weights(weightpath)
    
    for filename in os.listdir(realtestpath):
        print(filename)
        #加载数据
        print("Loading data...")
        f = open(realtestpath+filename, "rb")
        realdata = pickle.load(f, encoding="latin1")
        f.close()
    
        #进行预测和判定
        labels = model.predict(x = realdata[0],batch_size = 1)
        for i in range(len(labels)):
            if labels[i][0] >= 0.5:
                print(realdata[1][i])
'''

if __name__ == "__main__":
    batchSize = 64
    vectorDim = 30 #单个样本的单个token的向量维度
    maxLen = 900  #单个样本的切片长度
    dropout = 0.4  #模型dropout的比例
    traindataSetPath = "./data/dl_input/nvd/train_NVD/"  #训练数据集路径
    testdataSetPath = "./data/dl_input/nvd/test/"  #测试集路径
    #realtestdataSetPath = "../data_preprocess/data/realdata/aftercut/"  
    weightPath = 'model/bgru_0.4_k=1_0514.h5'  #保存模型的路径
    resultPath = "./BGRU/result/bgru_0.4_k=1_0514.2"  #测试结果保存路径
    #dealrawdata(raw_traindataSetPath, raw_testdataSetPath, traindataSetPath, testdataSetPath, batchSize, maxLen, vectorDim)
    main(traindataSetPath, testdataSetPath, weightPath, resultPath, batchSize, maxLen, vectorDim, dropout=dropout)
    #testrealdata(realtestdataSetPath, weightPath, batchSize, maxLen, vectorDim, dropout)#batchsize是输入模型的mini_batch的大小
