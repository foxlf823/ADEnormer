from os import listdir
from os.path import isfile, join
import os
import shutil
import bioc
from data_structure import Entity
import time
from metric import get_ner_fmeasure
import torch
import torch.autograd as autograd
import torch.nn as nn
import codecs
import logging
from options import opt
import numpy as np
import random


def setList(listt, value):
    if (value not in listt) and (value != u""):
        listt.append(value)
    return listt

def setMap(keyValueListMap, key, value):
    valueList = keyValueListMap.get(key)
    if valueList == None:
        valueList = list()
        keyValueListMap[key] = valueList
    valueList = setList(valueList, value)
    return keyValueListMap



def batchify_with_label(data, input_batch_list, input_batch_list_text, gpu):

    with torch.no_grad(): # feili, compatible with 0.4
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        if input_batch_list_text is None:
            chars = [sent[1] for sent in input_batch_list]
        if data.feat_config is not None:
            if len(input_batch_list[0]) > 3:
                labels = [sent[2] for sent in input_batch_list]
                features = [np.asarray(sent[3]) for sent in input_batch_list]
                feature_num = len(features[0][0])
            else:
                labels = None
                features = [np.asarray(sent[2]) for sent in input_batch_list]
                feature_num = len(features[0][0])

        else:
            if len(input_batch_list[0]) > 2:
                labels = [sent[2] for sent in input_batch_list]
            else:
                labels = None
        word_seq_lengths = torch.LongTensor(list(map(len, words)))

        if input_batch_list_text is not None:
            if labels:
                words_text = [sent[3] for sent in input_batch_list_text]
            else :
                words_text = [sent[2] for sent in input_batch_list_text]

        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len), dtype=torch.long))
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len), dtype=torch.long))
        if data.feat_config is not None:
            feature_seq_tensors = []
            for idx in range(feature_num):
                feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len), dtype=torch.long)))
        if input_batch_list_text is not None:
            words_text_tensor = [['<pad>' for col in range(max_seq_len)] for row in range(batch_size)]

        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len), dtype=torch.uint8))
        if labels:
            for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
                word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
                mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
                if data.feat_config is not None:
                    for idy in range(feature_num):
                        feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
                if input_batch_list_text is not None:
                    words_text_tensor[idx][:seqlen] = words_text[idx]

        else:
            for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
                word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
                if data.feat_config is not None:
                    for idy in range(feature_num):
                        feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
                if input_batch_list_text is not None:
                    words_text_tensor[idx][:seqlen] = words_text[idx]

        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        if data.feat_config is not None:
            for idx in range(feature_num):
                feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        if labels:
            label_seq_tensor = label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]

        if input_batch_list_text is not None:
            words_text_tensor_1 = []
            for i in range(batch_size):
                ii = word_perm_idx[i].item()
                words_text_tensor_1.append(words_text_tensor[ii])

            char_seq_tensor = None
            char_seq_lengths = None
            char_seq_recover = None
        else:
            words_text_tensor_1 = None
            ### deal with char
            # pad_chars (batch_size, max_seq_len)
            pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
            length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
            max_word_len = max(list(map(max, length_list)))
            char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long))
            char_seq_lengths = torch.LongTensor(length_list)
            for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
                for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                    # print len(word), wordlen
                    char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

            char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
            char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
            char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
            char_seq_tensor = char_seq_tensor[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)


        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        if opt.gpu >= 0 and torch.cuda.is_available():
            word_seq_tensor = word_seq_tensor.cuda(gpu)

            word_seq_lengths = word_seq_lengths.cuda(gpu)
            word_seq_recover = word_seq_recover.cuda(gpu)
            if labels:
                label_seq_tensor = label_seq_tensor.cuda(gpu)
            if data.feat_config is not None:
                for idx in range(feature_num):
                    feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda(gpu)
            if input_batch_list_text is None:
                char_seq_tensor = char_seq_tensor.cuda(gpu)
                char_seq_recover = char_seq_recover.cuda(gpu)
            mask = mask.cuda(gpu)

        if labels:
            if data.feat_config is not None:
                return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, feature_seq_tensors, words_text_tensor_1
            else:
                return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, None, words_text_tensor_1
        else:
            if data.feat_config is not None:
                return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, None, mask, feature_seq_tensors, words_text_tensor_1
            else:
                return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, None, mask, None, words_text_tensor_1


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    # print pred_variable.size()
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    if gold_variable is not None:
        gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    if gold_variable is not None:
        gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    if gold_variable is not None:
        gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        if gold_variable is not None:
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
        # print "g:", gold, gold_tag.tolist()

        pred_label.append(pred)
        if gold_variable is not None:
            gold_label.append(gold)
    if gold_variable is not None:
        return pred_label, gold_label
    else:
        return pred_label, None


def evaluate(data, opt, model, name, bEval, nbest=0):
    if name == "train":
        instances = data.train_Ids
        instances_text = data.train_texts
    elif name == "dev":
        instances = data.dev_Ids
        instances_text = data.dev_texts
    elif name == 'test':
        instances = data.test_Ids
        instances_text = data.test_texts
    else:
        logging.error("wrong evaluate name, {}".format(name))
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = opt.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if opt.elmo:
            instance_text = instances_text[start:end]
        else:
            instance_text = None
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_features, batch_text  \
            = batchify_with_label(data, instance, instance_text, opt.gpu)
        if nbest>0:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest, batch_features, batch_text)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, batch_features, batch_text)
        # print "tag:",tag_seq
        if bEval:
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label
            gold_results += gold_label
        else:
            pred_label, _ = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label

    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    if bEval:
        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, opt.schema)
    else:
        acc, p, r, f = None, None, None, None
    # if nbest>0:
    #     return speed, acc, p, r, f, nbest_pred_results, pred_scores
    # return speed, acc, p, r, f, pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores




def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True

def get_text_file(filename):
    file = codecs.open(filename, 'r', 'UTF-8')
    data = file.read()
    file.close()
    return data

# bioc will try to use str even if you feed it with utf-8.
# if bioc can't use str to denote something, it will use unicode
def get_bioc_file(filename):
    with codecs.open(filename, 'r', 'UTF-8') as fp:
        data = fp.read()
        collection = bioc.loads(data)
        return collection.documents

    # list_result = []
    # with bioc.iterparse(filename) as parser:
    #     for document in parser:
    #         list_result.append(document)
    # return list_result

def is_overlapped(a, b):
    if ((a.start <= b.start and a.end > b.start) or (a.start < b.end and a.end >= b.end) or
        (a.start >= b.start and a.end <= b.end) or (a.start <= b.start and a.end >= b.end)):
        return True
    else:
        return False

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def read_one_file(fileName, annotation_dir, entities_overlapped_types):
    annotation_file = get_bioc_file(join(annotation_dir, fileName))

    bioc_passage = annotation_file[0].passages[0]

    entities = []


    for entity in bioc_passage.annotations:
        entity_ = Entity()
        entity_.create(entity.id, entity.infons['type'], entity.locations[0].offset, entity.locations[0].end,
                       entity.text, None, None, None)

        for old_entity in entities:
            if is_overlapped(entity_, old_entity):

                logging.debug("entity overlapped: doc:{}, entity1_id:{}, entity1_type:{}, entity1_span:{} {}, entity2_id:{}, entity2_type:{}, entity2_span:{} {}"
                          .format(fileName, old_entity.id, old_entity.type, old_entity.start, old_entity.end,
                                  entity_.id, entity_.type, entity_.start, entity_.end))

                overlapped_types = entity_.type+"_"+old_entity.type if cmp(entity_.type, old_entity.type)>0 else old_entity.type+"_"+entity_.type
                if overlapped_types in entities_overlapped_types:
                    count = entities_overlapped_types[overlapped_types]
                    count += 1
                    entities_overlapped_types[overlapped_types] = count
                else:
                    entities_overlapped_types[overlapped_types] = 1


        entities.append(entity_)


def stat_entity_overlap(annotation_dir):
    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    entities_overlapped_types = {}

    for fileName in annotation_files:
        read_one_file(fileName, annotation_dir, entities_overlapped_types)

    print(entities_overlapped_types)

def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

def shuffle(a,b):
    assert len(a) == len(b)
    start_state = random.getstate()
    random.shuffle(a)
    random.setstate(start_state)
    random.shuffle(b)

# determine whether two spans are overlapped
def is_overlapped(a_start, a_end, b_start, b_end):
    if a_start <= b_start and a_end > b_start:
        return True
    elif a_start < b_end and a_end >= b_end:
        return True
    elif a_start >= b_start and a_end <= b_end:
        return True
    elif a_start <= b_start and a_end >= b_end:
        return True
    else:
        return False

def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.zeros([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb

# print("stat entity overlapped in MADE .........")
# stat_entity_overlap("/Users/feili/Desktop/umass/MADE/MADE-1.0/annotations")
# print("stat entity overlapped in Cardio .........")
# stat_entity_overlap('/Users/feili/Desktop/umass/bioC_data/Cardio_train/annotations')

#stat_entity_overlap('/Users/feili/Desktop/umass/CancerADE_SnoM_30Oct2017/bioc')
