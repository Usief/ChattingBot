from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk

import random
import pickle


padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    # batch, 包括encoder输入， decoder输入， decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

def load_dataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，文件包括word2id（单词与索引对应的字典）, id2word（单词与索引对应的反序字典），
                    training_samples样本数据，每一条是一个QA对
    :return: word2id, id2word, training_samples
    '''

    dataset_path = os.path.join(filename)
    print ('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        training_samples = data['trainingSamples']

    return word2id, id2word, training_samples

def create_batch(samples, en_de_seq_len):
    '''
    将batch进行padding, 并构造成placeholder所需要的格式
    :param samples: 一个batch的样本数据，列表，[question, answer]
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列最大长度
    :return: batch，可使用feed_dict传入模型进行训练
    '''

    # 格式化数据，使samples能够通过feed_dict传入session

    batch = Batch()
    # 根据样本长度获得batch size大小
    batch_size = len(samples)

    for i in range(batch_size):
        sample = samples[i]
        batch.encoderSeqs.append(list(reversed(sample[0])))     # 将输入反序，可提高模型效果。待验证
        batch.decoderSeqs.append([goToken] + sample[1] + [eosToken]) # Add  the ,go. and ,eos/ tokens
        batch.targetSeqs.append(batch.decoderSeqs[-1][1:]) # Same as decoder, but shifted  to the left(ignore the ,go>)

        # 将每个元素PAD到制定长度，并构造weights序列长度mask标志
        batch.encoderSeqs[i] = [padToken] * (en_de_seq_len[0] - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]
        batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (en_de_seq_len[1] - len(batch.targetSeqs[i])))
        batch.decoderSeqs[i] = batch.decoderSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.decoderSeqs[i]))
        batch.targetSeqs[i] = batch.targetSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.targetSeqs[i]))

    #--------------------将数据reshape, 编程序列长度*batch_size格式的数据----------------------------------------
    encoderSeqsT = [] # Corrected orientation
    for i in range(en_de_seq_len[0]):
        encoderSeqT = []
        for j in range(batch_size):
            encoderSeqT.append(batch.encoderSeqs[j][i])
        encoderSeqsT.append(encoderSeqT)
    batch.encoderSeqs = encoderSeqsT

    decoderSeqsT = []
    targetSeqsT = []
    weightsT = []
    for i in range(en_de_seq_len[1]):
        decoderSeqT = []
        targetSeqT = []
        weightT = []
        for j in range(batch_size):
            decoderSeqT.append(batch.decoderSeqs[j][i])
            targetSeqT.append(batch.targetSeqs[j][i])
            weightT.append(batch.weights[j][i])

        decoderSeqsT.append(decoderSeqT)
        targetSeqsT.append(targetSeqT)
        weightsT.append(weightT)

    batch.decoderSeqs = decoderSeqsT
    batch.targetSeqs = targetSeqsT
    batch.weights = weightsT

    return batch


def get_batches(data, batch_size, en_de_seq_len):
    '''
    对load_data()读出的数据进行处理，根据batch_size将原始数据分成不同大小的batch。然后对每个batchj当做参数传入create_batch进行格式处理
    :param data: load_data()读入的训练集，也就是QA对
    :param batch_size: batch的大小
    :param en_de_seq_len: 1*2的列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大元素
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''

    #每个epoch之前都需要对样本进行shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def gen_next_samples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in gen_next_samples():
        batch = create_batch(samples, en_de_seq_len)
        batches.append(batch)

    return batches

def sentence2enco(sentence, word2id, en_de_seq_len):


    if sentence == '':
        return None
    # 分词
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > en_de_seq_len[0]:
        return None

    # 将每个单词转化为id
    wordIds = []

    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))

    # 调用create_batch构造batch
    batch = create_batch([[wordIds,[]]], en_de_seq_len)

    return batch
