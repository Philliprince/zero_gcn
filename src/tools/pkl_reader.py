import pickle as pk
import numpy as np
import os
import json

# 转化零样本词向量文件
def load_word_embedding_list(file_dir):
    label_name = []
    word_list = np.zeros((230, 300))
    f = open(file_dir, 'rb')
    for i, line in enumerate(f.readlines()):
        arr = line.decode()
        arr = arr.strip().split()
        label_name.append(arr[0])
        word_list[i] = arr[1:]
    print(word_list)
    print(label_name)
    with open('../../data/word2vec.pkl', 'wb') as fv:
        pk.dump(word_list, fv)

# def generate_correspond_json(file_dir):
    # inv_wordn_file = os.path.join(data_dir, 'invdict_wordn.json')
    # with open(inv_wordn_file) as fp:
    #     json_data = json.load(fp)
    # seen_file = os.path.join(data_dir, '1k.txt')
    # unseen_file = os.path.join(data_dir, '%s.txt' % name)
    # seen_dict = {}
    # unseen_dict = {}
    # with open(seen_file) as fp:
    #     cnt = 0
    #     for line in fp:
    #         seen_dict[line.strip()] = cnt
    #         cnt += 1
    # with open(unseen_file) as fp:
    #     cnt = len(seen_dict)
    #     for line in fp:
    #         unseen_dict[line.strip()] = cnt
    #         cnt += 1

    # corresp_list= []
    # for i in range(190):
    #     corresp_list.append([i, 0])
    # for i in range(40):
    #     corresp_list.append([i, 1])
    # with open('../../data/list/corresp-zero.json', 'w') as fp:
    #     json.dump(corresp_list, fp)

data_dir = '../../data/'

if __name__ == '__main__':
    # load_word_embedding_list('../../data/class_wordembeddings.txt')

    # corresp_list= []
    # for i in range(1, 191):
    #     corresp_list.append([i, 0])
    # for i in range(191, 231):
    #     corresp_list.append([i, 1])
    # with open('../../data/list/corresp-zero.json', 'w') as fp:
    #     json.dump(corresp_list, fp)

    # with open('../../data/list/words.pkl', 'rb') as f:
    #     wv = pk.load(f)
    # print(wv)

    with open('../../data/glove_word2vec_wordnet.pkl', 'rb') as f:

        wv = pk.load(f, encoding='bytes')
    print(wv)

    # corresp_file = os.path.join(data_dir, 'list/corresp-all.json')  # 2-hops, 3-hops are also okay.
    # with open(corresp_file) as fp:
    #     corresp_list = json.load(fp)
    # print(corresp_list)