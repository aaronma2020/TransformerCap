# -*- coding: utf-8
import numpy as np


class Vocabulary():
    '''文本的单词表'''

    def __init__(self):
        self._word2id = {}
        self._id2word = {}
        self._idx = 0
        self._word = []

        # 特殊符号
        self.pad = '<pad>'
        self.bos = '<bos>'
        self.eos = '<eos>'
        self.unk = '<unk>'

        self.add_spe_sign()

    def add_word(self, word):
        '''添加单词'''
        if word not in self._word:
            self._word2id.update({word: self._idx})
            self._id2word.update({self._idx: word})
            self._word.append(word)
            self._idx += 1

    def word_to_id(self, word):
        '''把word转换成id的形式'''
        if word in self._word:
            return self._word2id[word]
        else:
            return self._word2id['<unk>']

    def id_to_word(self, id):
        '''把id的形式转换成word'''
        assert id <= self._idx, "输入的word id 大于最大的word id"
        return self._id2word[id]

    def tokenList_to_idList(self, tokenList, fixed_len):
        '''把tokenList转换成id的形式，，同时添加上<bos>，<eos>和<pad>
        :param tokenList: 包含一个句子的token形式， 如 ["a", "child", "holding", "a", "flowered", "umbrella", "and", "petting", "a", "yak"]
        :param fixed_len: 句子的最大长度，不包括<bos>和<eos>
        :return: list
        '''

        sent_len = len(tokenList)
        tok_id = [self.word_to_id(token) for token in tokenList]
        if sent_len < fixed_len:
            tok_id.insert(0, self._word2id[self.bos])
            tok_id.append(self._word2id[self.eos])
            pad_num = fixed_len - sent_len
            tok_id += [0] * pad_num
        else:
            tok_id = tok_id[:fixed_len]
            tok_id.insert(0, self._word2id[self.bos])
            tok_id.append(self._word2id[self.eos])
            sent_len = fixed_len

        return tok_id, sent_len

    def idList_to_sent(self, id_List):
        '''把idList转换成sent的形式

        :param id_List: 包含一个句子的id形式，如: [1, 4, 5, 343, 4, 123, 2389 ,213, 233 ,678 ,2343 ,2, 0, 0, 0, 0, 0, 0]
                        支持格式,: list, tensor, numpy.array
        :return: 一个句子的，如: 'A child holding a flowered umbrella and petting a yak'
        '''

        id_List = np.array(list(map(int, id_List)))
        word_array = np.array(self._word)
        eos_id = self._word2id[self.eos]
        eos_pos = np.where(id_List == eos_id)[0]
        if len(eos_pos >= 0):
            sent = word_array[id_List[1:eos_pos[0]]]
        else:
            sent = word_array[id_List[1:]]

        return ' '.join(sent)

    def add_spe_sign(self):
        self.add_word(self.pad)
        self.add_word(self.bos)
        self.add_word(self.eos)
        self.add_word(self.unk)

    def get_size(self):
        return self._idx

# if __name__ == '__main__':

# 功能测试

# vocab = Vocabulary()
# tokenList = ["a", "child", "holding", "a", "flowered", "umbrella", "and", "petting", "a", "yak"]
# idList = [1, 4, 5, 6, 4, 7, 8, 9, 10, 4, 11, 2, 0, 0, 0, 0, 0, 0]

# ******添加单词******

# for token in tokenList:
#     vocab.add_word(token)
# print('vocab__word2id:',vocab._word2id)
# print('vocab._id2word:', vocab._id2word)
# print('vocab length:', vocab.get_size())

# ('vocab__word2id:',
#  {'a': 4, 'and': 9, 'umbrella': 8, 'yak': 11, '<bos>': 1, 'flowered': 7, '<eos>': 2, '<pad>': 0, 'holding': 6,
#   'child': 5, 'petting': 10, '<unk>': 3})
# ('vocab._id2word:',
#  {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>', 4: 'a', 5: 'child', 6: 'holding', 7: 'flowered', 8: 'umbrella',
#   9: 'and', 10: 'petting', 11: 'yak'})
# ('vocab length:', 12)

# ******tokenList转换成idList******

# for token in tokenList:
#     vocab.add_word(token)
# idList = vocab.tokenList_to_idList(tokenList, 16)
# print('idList:', idList)

# ('idList:', [1, 4, 5, 6, 4, 7, 8, 9, 10, 4, 11, 2, 0, 0, 0, 0, 0, 0])

# ******idList转换成sent******

# for token in tokenList:
#     vocab.add_word(token)
#
# sent = vocab.idList_to_sent(idList)
# print('sent:', sent)
# ('sent:', 'a child holding a flowered umbrella and petting a yak')('sent:', 'a child holding a flowered umbrella and petting a yak')
