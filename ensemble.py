
import logging
import torch.nn as nn
from alphabet import Alphabet
from options import opt
import norm_utils
from data import build_pretrain_embedding, my_tokenize, load_data_fda
from my_utils import random_embedding, freeze_net
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import time
import os
from data_structure import Entity
import torch.nn.functional as functional
import math
import multi_sieve
import vsm
import norm_neural
import copy
from collections import Counter

class Ensemble(nn.Module):

    def __init__(self, word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses):
        super(Ensemble, self).__init__()

        self.word_alphabet = word_alphabet
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.dict_alphabet = dict_alphabet
        self.gpu = opt.gpu
        self.poses = poses
        self.dict_size = norm_utils.get_dict_size(dict_alphabet)

        self.vsm_linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.vsm_linear.weight.data.copy_(torch.eye(self.embedding_dim))

        self.neural_linear = nn.Linear(self.embedding_dim, self.dict_size, bias=False)

        # self.hidden_size = 2500
        # self.dropout = nn.Dropout(opt.dropout)
        # self.hidden = nn.Linear(3*self.dict_size, self.hidden_size)
        # self.relu = nn.ReLU()
        #
        # self.output = nn.Linear(self.hidden_size, self.dict_size)

        self.output = nn.Linear(3*self.dict_size, self.dict_size)

        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.vsm_linear = self.vsm_linear.cuda(self.gpu)
            self.neural_linear = self.neural_linear.cuda(self.gpu)

            # self.hidden = self.hidden.cuda(self.gpu)
            self.output = self.output.cuda(self.gpu)


        # if torch.cuda.is_available():
        #     self.w1 = torch.nn.Parameter(torch.tensor([0.3]).cuda(self.gpu))
        #     self.w2 = torch.nn.Parameter(torch.tensor([0.4]).cuda(self.gpu))
        #     self.w3 = torch.nn.Parameter(torch.tensor([0.3]).cuda(self.gpu))
        # else:
        #     self.w1 = torch.nn.Parameter(torch.tensor([0.3]))
        #     self.w2 = torch.nn.Parameter(torch.tensor([0.4]))
        #     self.w3 = torch.nn.Parameter(torch.tensor([0.3]))

    def forward(self, words, rules, lengths):

        length = words.size(1)
        mention_word_emb = self.word_embedding(words)
        mention_word_emb = mention_word_emb.unsqueeze_(1)
        mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1))
        mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1)

        length = self.poses.size(1)
        pos_word_emb = self.word_embedding(self.poses)
        pos_word_emb = pos_word_emb.unsqueeze_(1)
        pos_word_pool = functional.avg_pool2d(pos_word_emb, (length, 1))
        pos_word_pool = pos_word_pool.squeeze_(1).squeeze_(1)

        m_W = self.vsm_linear(mention_word_pool)
        vsm_confidences = torch.matmul(m_W, torch.t(pos_word_pool))
        vsm_confidences = functional.softmax(vsm_confidences, dim=1)

        # batch_size = words.size(0)
        # rule_confidences = torch.zeros(batch_size, self.dict_size)
        # if torch.cuda.is_available():
        #     rule_confidences = rule_confidences.cuda(self.gpu)
        # rule_confidences = rule_confidences.scatter_(1, rules, 1)
        rule_confidences = rules

        neural_confidences = self.neural_linear(mention_word_pool)
        neural_confidences = functional.softmax(neural_confidences, dim=1)

        # confidences = self.w1*rule_confidences+self.w2*vsm_confidences+self.w3*neural_confidences

        # confidences = self.w1 * rule_confidences + self.w2 * vsm_confidences \
        #               + self.w3 * neural_confidences

        x = torch.cat((rule_confidences, vsm_confidences, neural_confidences), 1)
        # x = self.relu(self.hidden(self.dropout(x)))
        confidences = self.output(x)

        return confidences


    def loss(self, y_pred, y_gold):
        return self.criterion(y_pred, y_gold)

    # def normalize(self):
    #
    #     e_w1 = torch.exp(self.w1.data)
    #     e_w2 = torch.exp(self.w2.data)
    #     e_w3 = torch.exp(self.w3.data)
    #     e = e_w1+e_w2+e_w3+1e-8
    #
    #     self.w1.data = e_w1 / e
    #     self.w2.data = e_w2 / e
    #     self.w3.data = e_w3 / e

        # min = 99999
        # max = -99999
        # for x in [self.w1, self.w2, self.w3]:
        #     if x < min:
        #         min = x.data
        #     if x > max:
        #         max = x.data
        #
        # self.w1.data = (self.w1.data-min)/(max-min)
        # self.w2.data = (self.w2.data- min) / (max - min)
        # self.w3.data = (self.w3.data- min) / (max - min)

    def process_one_doc(self, doc, entities, dictionary, dictionary_reverse, isMeddra_dict):

        Xs, Ys = generate_instances(doc, self.word_alphabet, self.dict_alphabet, dictionary, dictionary_reverse, isMeddra_dict)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            words, rules, lengths, _ = next(data_iter)

            y_pred = self.forward(words, rules, lengths)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = lengths.size(0)

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = norm_utils.get_dict_name(self.dict_alphabet, indices[batch_idx].item())
                if isMeddra_dict:
                    name = dictionary[norm_id]
                    entity.norm_ids.append(norm_id)
                    entity.norm_names.append(name)
                else:
                    concept = dictionary[norm_id]
                    entity.norm_ids.append(norm_id)
                    entity.norm_names.append(concept.names)

            entity_start += actual_batch_size

def generate_instances(document, word_alphabet, dict_alphabet, dictionary, dictionary_reverse, isMeddra_dict):
    Xs = []
    Ys = []

    # copy entities from gold entities
    pred_entities = []
    for gold in document.entities:
        pred = Entity()
        pred.id = gold.id
        pred.type = gold.type
        pred.spans = gold.spans
        pred.section = gold.section
        pred.name = gold.name
        pred_entities.append(pred)

    multi_sieve.runMultiPassSieve(document, pred_entities, dictionary, isMeddra_dict)

    for idx, entity in enumerate(document.entities):

        if isMeddra_dict:
            if len(entity.norm_ids) > 0:
                Y = norm_utils.get_dict_index(dict_alphabet, entity.norm_ids[0])
                if Y >= 0 and Y < norm_utils.get_dict_size(dict_alphabet):
                    Ys.append(Y)
                else:
                    continue
            else:
                Ys.append(0)
        else:
            if len(entity.norm_ids) > 0:
                if entity.norm_ids[0] in dictionary_reverse:
                    cui_list = dictionary_reverse[entity.norm_ids[0]]
                    Y = norm_utils.get_dict_index(dict_alphabet, cui_list[0])  # use the first id to generate instance
                    if Y >= 0 and Y < norm_utils.get_dict_size(dict_alphabet):
                        Ys.append(Y)
                    else:
                        raise RuntimeError("entity {}, {}, cui not in dict_alphabet".format(entity.id, entity.name))
                else:
                    logging.info("entity {}, {}, can't map to umls, ignored".format(entity.id, entity.name))
                    continue
            else:
                Ys.append(0)

        X = dict()

        tokens = my_tokenize(entity.name)
        word_ids = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            word_ids.append(word_id)
        X['word'] = word_ids

        if pred_entities[idx].rule_id is None:
            X['rule'] = [0]*norm_utils.get_dict_size(dict_alphabet)
        else:
            X['rule'] = [0]*norm_utils.get_dict_size(dict_alphabet)
            X['rule'][norm_utils.get_dict_index(dict_alphabet, pred_entities[idx].rule_id)] = 1

        Xs.append(X)

    return Xs, Ys

class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def my_collate(batch):
    x, y = zip(*batch)

    words = [s['word'] for s in x]
    rules = [s['rule'] for s in x]

    lengths = [len(row) for row in words]
    max_len = max(lengths)

    words = pad_sequence(words, max_len)
    rules = torch.tensor(rules, dtype=torch.float32)

    lengths = torch.LongTensor(lengths)
    y = torch.LongTensor(y).view(-1)

    if opt.gpu >= 0 and torch.cuda.is_available():
        words = words.cuda(opt.gpu)
        rules = rules.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)

    return words, rules, lengths, y

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x

def train(train_data, dev_data, test_data, d, dictionary, dictionary_reverse, opt, fold_idx, isMeddra_dict):
    logging.info("train the ensemble normalization model ...")

    external_train_data = []
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                external_train_data.extend(load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, True))
            else:
                raise RuntimeError("not support external corpus")
    if len(external_train_data) != 0:
        train_data.extend(external_train_data)


    logging.info("build alphabet ...")
    word_alphabet = Alphabet('word')
    norm_utils.build_alphabet_from_dict(word_alphabet, dictionary, isMeddra_dict)
    norm_utils.build_alphabet(word_alphabet, train_data)
    if opt.dev_file:
        norm_utils.build_alphabet(word_alphabet, dev_data)
    if opt.test_file:
        norm_utils.build_alphabet(word_alphabet, test_data)
    norm_utils.fix_alphabet(word_alphabet)

    if d.config.get('norm_emb') is not None:
        logging.info("load pretrained word embedding ...")
        pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_emb'),
                                                                         word_alphabet,
                                                                              opt.word_emb_dim, False)
        word_embedding = nn.Embedding(word_alphabet.size(), word_emb_dim, padding_idx=0)
        word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        embedding_dim = word_emb_dim
    else:
        logging.info("randomly initialize word embedding ...")
        word_embedding = nn.Embedding(word_alphabet.size(), d.word_emb_dim, padding_idx=0)
        word_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(word_alphabet.size(), d.word_emb_dim)))
        embedding_dim = d.word_emb_dim

    dict_alphabet = Alphabet('dict')
    norm_utils.init_dict_alphabet(dict_alphabet, dictionary)
    norm_utils.fix_alphabet(dict_alphabet)

    # rule
    logging.info("init rule-based normer")
    multi_sieve.init(opt, train_data, d, dictionary, dictionary_reverse, isMeddra_dict)

    if opt.ensemble == 'learn':
        logging.info("init ensemble normer")
        poses = vsm.init_vector_for_dict(word_alphabet, dict_alphabet, dictionary, isMeddra_dict)
        ensemble_model = Ensemble(word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses)
        if pretrain_neural_model is not None:
            ensemble_model.neural_linear.weight.data.copy_(pretrain_neural_model.linear.weight.data)
        if pretrain_vsm_model is not None:
            ensemble_model.vsm_linear.weight.data.copy_(pretrain_vsm_model.linear.weight.data)
        ensemble_train_X = []
        ensemble_train_Y = []
        for doc in train_data:
            temp_X, temp_Y = generate_instances(doc, word_alphabet, dict_alphabet, dictionary, dictionary_reverse, isMeddra_dict)

            ensemble_train_X.extend(temp_X)
            ensemble_train_Y.extend(temp_Y)
        ensemble_train_loader = DataLoader(MyDataset(ensemble_train_X, ensemble_train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)
        ensemble_optimizer = optim.Adam(ensemble_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        if opt.tune_wordemb == False:
            freeze_net(ensemble_model.word_embedding)
    else:

        # vsm
        logging.info("init vsm-based normer")
        poses = vsm.init_vector_for_dict(word_alphabet, dict_alphabet, dictionary, isMeddra_dict)
        # alphabet can share between vsm and neural since they don't change
        # but word_embedding cannot
        vsm_model = vsm.VsmNormer(word_alphabet, copy.deepcopy(word_embedding), embedding_dim, dict_alphabet, poses)
        vsm_train_X = []
        vsm_train_Y = []
        for doc in train_data:
            if isMeddra_dict:
                temp_X, temp_Y = vsm.generate_instances(doc.entities, word_alphabet, dict_alphabet)
            else:
                temp_X, temp_Y = vsm.generate_instances_ehr(doc.entities, word_alphabet, dict_alphabet, dictionary_reverse)

            vsm_train_X.extend(temp_X)
            vsm_train_Y.extend(temp_Y)
        vsm_train_loader = DataLoader(vsm.MyDataset(vsm_train_X, vsm_train_Y), opt.batch_size, shuffle=True, collate_fn=vsm.my_collate)
        vsm_optimizer = optim.Adam(vsm_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        if opt.tune_wordemb == False:
            freeze_net(vsm_model.word_embedding)

        if d.config['norm_vsm_pretrain'] == '1':
            vsm.dict_pretrain(dictionary, dictionary_reverse, d, True, vsm_optimizer, vsm_model)

        # neural
        logging.info("init neural-based normer")
        neural_model = norm_neural.NeuralNormer(word_alphabet, copy.deepcopy(word_embedding), embedding_dim, dict_alphabet)

        neural_train_X = []
        neural_train_Y = []
        for doc in train_data:
            if isMeddra_dict:
                temp_X, temp_Y = norm_neural.generate_instances(doc.entities, word_alphabet, dict_alphabet)
            else:
                temp_X, temp_Y = norm_neural.generate_instances_ehr(doc.entities, word_alphabet, dict_alphabet, dictionary_reverse)

            neural_train_X.extend(temp_X)
            neural_train_Y.extend(temp_Y)
        neural_train_loader = DataLoader(norm_neural.MyDataset(neural_train_X, neural_train_Y), opt.batch_size, shuffle=True, collate_fn=norm_neural.my_collate)
        neural_optimizer = optim.Adam(neural_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        if opt.tune_wordemb == False:
            freeze_net(neural_model.word_embedding)

        if d.config['norm_neural_pretrain'] == '1':
            neural_model.dict_pretrain(dictionary, dictionary_reverse, d, True, neural_optimizer, neural_model)


    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")
    for idx in range(opt.iter):
        epoch_start = time.time()

        if opt.ensemble == 'learn':

            ensemble_model.train()
            ensemble_train_iter = iter(ensemble_train_loader)
            ensemble_num_iter = len(ensemble_train_loader)

            for i in range(ensemble_num_iter):
                x, rules, lengths, y = next(ensemble_train_iter)

                y_pred = ensemble_model.forward(x, rules, lengths)

                l = ensemble_model.loss(y_pred, y)

                l.backward()

                if opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), opt.gradient_clip)
                ensemble_optimizer.step()
                ensemble_model.zero_grad()


        else:

            vsm_model.train()
            vsm_train_iter = iter(vsm_train_loader)
            vsm_num_iter = len(vsm_train_loader)

            for i in range(vsm_num_iter):
                x, lengths, y = next(vsm_train_iter)

                l, _ = vsm_model.forward_train(x, lengths, y)

                l.backward()

                if opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
                vsm_optimizer.step()
                vsm_model.zero_grad()


            neural_model.train()
            neural_train_iter = iter(neural_train_loader)
            neural_num_iter = len(neural_train_loader)

            for i in range(neural_num_iter):

                x, lengths, y = next(neural_train_iter)

                y_pred = neural_model.forward(x, lengths)

                l = neural_model.loss(y_pred, y)

                l.backward()

                if opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(neural_model.parameters(), opt.gradient_clip)
                neural_optimizer.step()
                neural_model.zero_grad()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))


        if opt.dev_file:
            if opt.ensemble == 'learn':
                # logging.info("weight w1: %.4f, w2: %.4f, w3: %.4f" % (ensemble_model.w1.data.item(), ensemble_model.w2.data.item(), ensemble_model.w3.data.item()))
                p, r, f = norm_utils.evaluate(dev_data, dictionary, dictionary_reverse, None, None, ensemble_model, d, isMeddra_dict)
            else:
                p, r, f = norm_utils.evaluate(dev_data, dictionary, dictionary_reverse, vsm_model, neural_model, None, d, isMeddra_dict)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            if opt.ensemble == 'learn':
                if fold_idx is None:
                    torch.save(ensemble_model, os.path.join(opt.output, "ensemble.pkl"))
                else:
                    torch.save(ensemble_model, os.path.join(opt.output, "ensemble_{}.pkl".format(fold_idx+1)))
            else:
                if fold_idx is None:
                    torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
                    torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))
                else:
                    torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx+1)))
                    torch.save(neural_model, os.path.join(opt.output, "norm_neural_{}.pkl".format(fold_idx + 1)))

            best_dev_f = f
            best_dev_p = p
            best_dev_r = r

            bad_counter = 0
        else:
            bad_counter += 1

        if len(opt.dev_file) != 0 and bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break


    logging.info("train finished")

    if fold_idx is None:
        multi_sieve.finalize(True)
    else:
        if fold_idx == opt.cross_validation-1:
            multi_sieve.finalize(True)
        else:
            multi_sieve.finalize(False)

    if len(opt.dev_file) == 0:
        if opt.ensemble == 'learn':
            torch.save(ensemble_model, os.path.join(opt.output, "ensemble.pkl"))
        else:
            torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
            torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))

    return best_dev_p, best_dev_r, best_dev_f


def merge_result(entities1, entities2, entities3, merge_entities, dictionary, isMeddra_dict, dict_alphabet, d):
    if opt.ensemble == 'vote':

        for idx, merge_entity in enumerate(merge_entities):
            entity1 = entities1[idx]
            entity2 = entities2[idx]
            entity3 = entities3[idx]

            if entity1.rule_id is None:
                if entity2.vsm_id == entity3.neural_id:
                    merge_entity.norm_ids.append(entity2.norm_ids[0])
                    merge_entity.norm_names.append(entity2.norm_names[0])
                else:
                    # if entity2.norm_confidences[0] >= entity3.norm_confidences[0]:
                    #     merge_entity.norm_ids.append(entity2.norm_ids[0])
                    #     merge_entity.norm_names.append(entity2.norm_names[0])
                    # else:
                    #     merge_entity.norm_ids.append(entity3.norm_ids[0])
                    #     merge_entity.norm_names.append(entity3.norm_names[0])

                    # vsm is prior to others
                    merge_entity.norm_ids.append(entity2.norm_ids[0])
                    merge_entity.norm_names.append(entity2.norm_names[0])

            else:

                id_and_ticket = Counter()
                id_and_ticket[entity1.norm_ids[0]] = id_and_ticket[entity1.norm_ids[0]] +1
                id_and_ticket[entity2.norm_ids[0]] = id_and_ticket[entity2.norm_ids[0]] +1
                id_and_ticket[entity3.norm_ids[0]] = id_and_ticket[entity3.norm_ids[0]] +1

                temp_id_name = {}
                temp_id_name[entity1.norm_ids[0]] = entity1.norm_names[0]
                temp_id_name[entity2.norm_ids[0]] = entity2.norm_names[0]
                temp_id_name[entity3.norm_ids[0]] = entity3.norm_names[0]

                top_id, top_ct = id_and_ticket.most_common(1)[0]
                if top_ct == 1:
                    # the confidence of rule is always 1
                    # merge_entity.norm_ids.append(entity1.norm_ids[0])
                    # merge_entity.norm_names.append(entity1.norm_names[0])

                    # vsm is prior to others
                    merge_entity.norm_ids.append(entity2.norm_ids[0])
                    merge_entity.norm_names.append(entity2.norm_names[0])
                else:
                    merge_entity.norm_ids.append(top_id)
                    merge_entity.norm_names.append(temp_id_name[top_id])

    elif opt.ensemble == 'sum':

        for idx, merge_entity in enumerate(merge_entities):
            entity1 = entities1[idx]
            entity2 = entities2[idx]
            entity3 = entities3[idx]

            if entity1.rule_id is None:

                total = float(d.config['norm_ensumble_sum_weight']['1']['w2'])*entity2.norm_confidences[0] + \
                        float(d.config['norm_ensumble_sum_weight']['1']['w3'])*entity3.norm_confidences[0]
            else:
                total = float(d.config['norm_ensumble_sum_weight']['2']['w1'])*entity1.norm_confidences[0] + \
                        float(d.config['norm_ensumble_sum_weight']['2']['w2'])*entity2.norm_confidences[0] + \
                        float(d.config['norm_ensumble_sum_weight']['2']['w3'])*entity3.norm_confidences[0]

            index = total.argmax()
            norm_id = norm_utils.get_dict_name(dict_alphabet, index)
            if isMeddra_dict:
                name = dictionary[norm_id]
                merge_entity.norm_ids.append(norm_id)
                merge_entity.norm_names.append(name)
            else:
                concept = dictionary[norm_id]
                merge_entity.norm_ids.append(norm_id)
                merge_entity.norm_names.append(concept.names)

    else:
        raise RuntimeError("run configuration")

    return merge_entities