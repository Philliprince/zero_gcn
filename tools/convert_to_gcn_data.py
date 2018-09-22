import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os
from tensorflow.python import pywrap_tensorflow

from config.configs import config


def convert_input(wv_file, save_dir):
    """

    :param wv_file:
    :param save_dir:
    :return:
    """
    with open(wv_file, 'rb') as fp:
        feats = pkl.load(fp, encoding='bytes')
    feats = feats.tolist()
    sparse_feats = sparse.csr_matrix(feats)
    dense_feats = np.array(feats)

    sparse_file = os.path.join(save_dir, 'ind.NELL.allx')
    dense_file = os.path.join(save_dir, 'ind.NELL.allx_dense')

    with open(sparse_file, 'wb') as fp:
        pkl.dump(sparse_feats, fp)
    with open(dense_file, 'wb') as fp:
        pkl.dump(dense_feats, fp)
    print('Save feat in shape to', sparse_file, dense_file, 'with shape', dense_feats.shape)
    return


def prepare_graph():
    pass


def convert_graph(save_dir):
    """
    如果不存在imagenet_graph.pkl就下载，存在就为imagenet_graph.pkl创建一个软链接
    :param save_dir:
    :return:
    """
    graph_file = os.path.join(config.exp, 'imagenet_graph.pkl')
    if not os.path.exists(graph_file):
        prepare_graph()
    save_file = os.path.join(save_dir, 'ind.NELL.graph')
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)
    # ln的功能是为某一个文件在另外一个位置建立一个不同的链接
    # 具体用法是：ln –s 源文件 目标文件。
    cmd = 'ln -s %s %s' % (graph_file, save_file)
    os.system(cmd)
    return


def convert_label(model_path, layer_name, save_dir, offset):
    corresp_file = os.path.join(config.exp, 'list/corresp-zero.json')  # 修改为零样本分类对应230类的标签
    with open(corresp_file) as fp:
        corresp_list = json.load(fp)

    # 读取ckpt模型checkpoint
    def get_variables_in_checkpoint_file(file_name):
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return reader, var_to_shape_map
    reader, var_keep_dic = get_variables_in_checkpoint_file(model_path)
    for name in var_keep_dic:
        print(name, len(var_keep_dic[name]), var_keep_dic[name])
        if name == layer_name:
            print(name + ':' + reader.get_tensor(name).shape)
            fc = reader.get_tensor(name).squeeze()
            fc_dim = fc.shape[0]
            break
    # 类似于one-hot编码机制？
    fc_labels = np.zeros((len(corresp_list), fc_dim))

    test_index = []
    for i in range(len(corresp_list)):
        class_id = corresp_list[i][0]
        if class_id == -1 or corresp_list[i][1] == 1:
            continue
        assert class_id < 1000
        fc_labels[i, :] = np.copy(fc[:, class_id + offset])

    for i in range(len(corresp_list)):
        if corresp_list[i][0] == -1:
            test_index.append(-1)
        else:
            test_index.append(corresp_list[i][1])

    label_file = os.path.join(save_dir, 'ind.NELL.ally_multi')
    test_file = os.path.join(save_dir, 'ind.NELL.index')
    with open(label_file, 'wb') as fp:
        pkl.dump(fc_labels, fp)
    with open(test_file, 'wb') as fp:
        pkl.dump(test_index, fp)
    return
    pass


def convert_to_gcn_data(model_path, layer_name, offset, wv_file):
    save_dir = os.path.join(config.exp, 'glove_res50')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # convert input
    convert_input(wv_file, save_dir)
    # convert graph
    convert_graph(save_dir)
    # convert label
    convert_label(model_path, layer_name, save_dir, offset)
