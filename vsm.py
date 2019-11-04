import torch.nn as nn
import logging
from alphabet import Alphabet
from my_utils import random_embedding, freeze_net
import torch
from data import build_pretrain_embedding, my_tokenize, load_data_fda
import numpy as np
import torch.nn.functional as functional
import os
from data_structure import Entity
import norm_utils
from options import opt
import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time
from norm_neural import DotAttentionLayer

class VsmNormer(nn.Module):

    def __init__(self, word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses, poses_lengths):
        super(VsmNormer, self).__init__()
        self.word_alphabet = word_alphabet
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.dict_alphabet = dict_alphabet
        self.gpu = opt.gpu
        self.poses = poses
        self.dict_size = norm_utils.get_dict_size(dict_alphabet)
        self.margin = 1
        self.poses_lengths = poses_lengths
        self.word_drop = nn.Dropout(opt.dropout)
        self.attn = DotAttentionLayer(self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear.weight.data.copy_(torch.eye(self.embedding_dim))

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.attn = self.attn.cuda(self.gpu)
            self.linear = self.linear.cuda(self.gpu)


    def forward_train(self, mention, lengths, y):

        # length = mention.size(1)
        mention_word_emb = self.word_embedding(mention)
        mention_word_emb = self.word_drop(mention_word_emb)
        # mention_word_emb = mention_word_emb.unsqueeze_(1)
        # mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1))
        # mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1)
        mention_word_pool = self.attn((mention_word_emb, lengths))

        # length = self.poses.size(1)
        pos_word_emb = self.word_embedding(self.poses)
        pos_word_emb = self.word_drop(pos_word_emb)
        # pos_word_emb = pos_word_emb.unsqueeze_(1)
        # pos_word_pool = functional.avg_pool2d(pos_word_emb, (length, 1))
        # pos_word_pool = pos_word_pool.squeeze_(1).squeeze_(1)
        pos_word_pool = self.attn((pos_word_emb, self.poses_lengths))

        m_W = self.linear(mention_word_pool)

        pos_similarity = torch.matmul(m_W, torch.t(pos_word_pool))


        y_expand = y.unsqueeze(-1).expand(-1, self.dict_size)


        a = torch.gather(pos_similarity, 1, y_expand)


        loss = (self.margin-a+pos_similarity).clamp(min=0)

        batch_size = mention.size(0)
        one_hot = torch.zeros(batch_size, self.dict_size)
        if opt.gpu >= 0 and torch.cuda.is_available():
            one_hot = one_hot.cuda(opt.gpu)

        one_hot = one_hot.scatter_(1, y.unsqueeze(-1), self.margin)

        loss = torch.sum(loss - one_hot) / batch_size

        # y = y.unsqueeze(-1).expand(-1, self.dict_size)
        #
        #
        # a = torch.gather(pos_similarity, 1, y)
        #
        #
        # loss = (1-a+pos_similarity).clamp(min=0)
        #
        # loss = torch.sum(loss)-1


        return loss, pos_similarity

    def forward_eval(self, mention, lengths):
        # length = mention.size(1)
        mention_word_emb = self.word_embedding(mention)
        mention_word_emb = self.word_drop(mention_word_emb)
        # mention_word_emb = mention_word_emb.unsqueeze_(1)
        # mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1))
        # mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1)
        mention_word_pool = self.attn((mention_word_emb, lengths))

        # length = self.poses.size(1)
        pos_word_emb = self.word_embedding(self.poses)
        pos_word_emb = self.word_drop(pos_word_emb)
        # pos_word_emb = pos_word_emb.unsqueeze_(1)
        # pos_word_pool = functional.avg_pool2d(pos_word_emb, (length, 1))
        # pos_word_pool = pos_word_pool.squeeze_(1).squeeze_(1)
        pos_word_pool = self.attn((pos_word_emb, self.poses_lengths))

        m_W = self.linear(mention_word_pool)

        pos_similarity = torch.matmul(m_W, torch.t(pos_word_pool))

        return pos_similarity

    def normalize(self, similarities):

        return functional.softmax(similarities, dim=1)

    def process_one_doc(self, doc, entities, dictionary, dictionary_reverse, isMeddra_dict):

        if isMeddra_dict:
            Xs, Ys = generate_instances(entities, self.word_alphabet, self.dict_alphabet)
        else:
            Xs, Ys = generate_instances_ehr(entities, self.word_alphabet, self.dict_alphabet, dictionary_reverse)


        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            mention, lengths, _ = next(data_iter)

            similarities = self.forward_eval(mention, lengths)

            similarities = self.normalize(similarities)

            values, indices = torch.max(similarities, 1)

            actual_batch_size = mention.size(0)

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

                if opt.ensemble == 'sum':
                    entity.norm_confidences.append(similarities[batch_idx].detach().cpu().numpy())
                else:
                    entity.norm_confidences.append(values[batch_idx].item())
                entity.vsm_id = norm_id

            entity_start += actual_batch_size




class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def generate_instances(entities, word_alphabet, dict_alphabet):
    Xs = []
    Ys = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)

    for entity in entities:
        if len(entity.norm_ids) > 0:
            Y = norm_utils.get_dict_index(dict_alphabet, entity.norm_ids[0])
            if Y >= 0 and Y < dict_size:
                pass
            else:
                continue
        else:
            Y = 0

        # mention
        tokens = my_tokenize(entity.name)
        mention = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            mention.append(word_id)

        Xs.append(mention)
        Ys.append(Y)


    return Xs, Ys

def generate_instances_ehr(entities, word_alphabet, dict_alphabet, dictionary_reverse):
    Xs = []
    Ys = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)

    for entity in entities:

        if len(entity.norm_ids) > 0:
            if entity.norm_ids[0] in dictionary_reverse:
                cui_list = dictionary_reverse[entity.norm_ids[0]]
                Y = norm_utils.get_dict_index(dict_alphabet, cui_list[0])  # use the first id to generate instance
                if Y >= 0 and Y < dict_size:
                    pass
                else:
                    raise RuntimeError("entity {}, {}, cui not in dict_alphabet".format(entity.id, entity.name))
            else:
                logging.debug("entity {}, {}, can't map to umls, ignored".format(entity.id, entity.name))
                continue
        else:
            Y = 0

        # mention
        tokens = my_tokenize(entity.name)
        mention = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            mention.append(word_id)

        Xs.append(mention)
        Ys.append(Y)


    return Xs, Ys


# def generate_instances_for_eval(entity, word_alphabet):
#
#     # mention
#     tokens = my_tokenize(entity.name)
#     mention = []
#     for token in tokens:
#         token = norm_utils.word_preprocess(token)
#         word_id = word_alphabet.get_index(token)
#         mention.append(word_id)
#     max_len = len(mention)
#     mentions = [mention]
#     mentions = pad_sequence(mentions, max_len)
#
#     if torch.cuda.is_available():
#         mentions = mentions.cuda(opt.gpu)
#
#     return mentions

def init_vector_for_dict(word_alphabet, dict_alphabet, dictionary, isMeddra_dict):

    # pos
    poses = []
    poses_lengths = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)
    max_len = 0
    for i in range(dict_size):

        # pos
        if isMeddra_dict:
            concept_name = dictionary[norm_utils.get_dict_name(dict_alphabet, i)]
            tokens = my_tokenize(concept_name)
        else:
            concept = dictionary[norm_utils.get_dict_name(dict_alphabet, i)]
            tokens = my_tokenize(concept.names[0])
        pos = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            pos.append(word_id)

        if len(pos) > max_len:
            max_len = len(pos)

        poses.append(pos)
        poses_lengths.append(len(pos))

    poses = pad_sequence(poses, max_len)
    poses_lengths = torch.LongTensor(poses_lengths)

    if opt.gpu >= 0 and torch.cuda.is_available():
        poses = poses.cuda(opt.gpu)
        poses_lengths = poses_lengths.cuda(opt.gpu)

    return poses, poses_lengths


def my_collate(batch):
    x, y = zip(*batch)

    lengths = [len(row) for row in x]
    max_len = max(lengths)
    x = pad_sequence(x, max_len)

    lengths = torch.LongTensor(lengths)
    y = torch.LongTensor(y).view(-1)


    if opt.gpu >= 0 and torch.cuda.is_available():
        x = x.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)

    return x, lengths, y

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x


def generate_dict_instances(dictionary, dict_alphabet, word_alphabet, isMeddra_dict):
    Xs = []
    Ys = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)

    if isMeddra_dict:
        for concept_id, concept_name in dictionary.items():

            Y = norm_utils.get_dict_index(dict_alphabet, concept_id)
            if Y >= 0 and Y < norm_utils.get_dict_size(dict_alphabet):
                Ys.append(Y)
            else:
                continue


            tokens = my_tokenize(concept_name)
            word_ids = []
            for token in tokens:
                token = norm_utils.word_preprocess(token)
                word_id = word_alphabet.get_index(token)
                word_ids.append(word_id)

            Xs.append(word_ids)
    else:
        for concept_id, concept in dictionary.items():
            Y = norm_utils.get_dict_index(dict_alphabet, concept_id)
            if Y >= 0 and Y < norm_utils.get_dict_size(dict_alphabet):
                pass
            else:
                continue

            # for concept_name in concept.names:
            #
            #     tokens = my_tokenize(concept_name)
            #     word_ids = []
            #     for token in tokens:
            #         token = norm_utils.word_preprocess(token)
            #         word_id = word_alphabet.get_index(token)
            #         word_ids.append(word_id)
            #
            #     Ys.append(Y)
            #     Xs.append(word_ids)


            tokens = my_tokenize(concept.names[0])
            word_ids = []
            for token in tokens:
                token = norm_utils.word_preprocess(token)
                word_id = word_alphabet.get_index(token)
                word_ids.append(word_id)

            Ys.append(Y)
            Xs.append(word_ids)

    return Xs, Ys





def dict_pretrain(dictionary, dictionary_reverse, d, isMeddra_dict, optimizer, vsm_model):
    logging.info('use dict pretrain ...')

    dict_Xs, dict_Ys = generate_dict_instances(dictionary, vsm_model.dict_alphabet, vsm_model.word_alphabet, isMeddra_dict)

    data_loader = DataLoader(MyDataset(dict_Xs, dict_Ys), opt.batch_size, shuffle=True, collate_fn=my_collate)

    expected_accuracy = int(d.config['norm_vsm_pretrain_accuracy'])

    logging.info("start dict pretraining ...")

    bad_counter = 0
    best_accuracy = 0

    for idx in range(9999):
        epoch_start = time.time()

        vsm_model.train()

        correct, total = 0, 0

        sum_loss = 0

        train_iter = iter(data_loader)
        num_iter = len(data_loader)

        for i in range(num_iter):

            x, lengths, y = next(train_iter)

            l, y_pred = vsm_model.forward_train(x, lengths, y)

            sum_loss += l.item()

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
            optimizer.step()
            vsm_model.zero_grad()

            total += y.size(0)
            _, pred = torch.max(y_pred, 1)
            correct += (pred == y).sum().item()

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s pretraining finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
            idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

        if accuracy > expected_accuracy:

            logging.info("Exceed {}% training accuracy, breaking ... ".format(expected_accuracy))
            break

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    return


def train(train_data, dev_data, test_data, d, dictionary, dictionary_reverse, opt, fold_idx, isMeddra_dict):
    logging.info("train the vsm-based normalization model ...")

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
    logging.info("alphabet size {}".format(word_alphabet.size()))

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

    logging.info("init_vector_for_dict")
    poses, poses_lengths = init_vector_for_dict(word_alphabet, dict_alphabet, dictionary, isMeddra_dict)


    vsm_model = VsmNormer(word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses, poses_lengths)


    logging.info("generate instances for training ...")
    train_X = []
    train_Y = []

    for doc in train_data:
        if isMeddra_dict:
            temp_X, temp_Y = generate_instances(doc.entities, word_alphabet, dict_alphabet)
        else:
            temp_X, temp_Y = generate_instances_ehr(doc.entities, word_alphabet, dict_alphabet, dictionary_reverse)
        train_X.extend(temp_X)
        train_Y.extend(temp_Y)


    train_loader = DataLoader(MyDataset(train_X, train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)

    optimizer = optim.Adam(vsm_model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    if opt.tune_wordemb == False:
        freeze_net(vsm_model.word_embedding)


    if d.config['norm_vsm_pretrain'] == '1':
        dict_pretrain(dictionary, dictionary_reverse, d, isMeddra_dict, optimizer, vsm_model)


    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        vsm_model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        sum_loss = 0

        correct, total = 0, 0

        for i in range(num_iter):

            x, lengths, y = next(train_iter)

            l, y_pred = vsm_model.forward_train(x, lengths, y)

            sum_loss += l.item()

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
            optimizer.step()
            vsm_model.zero_grad()

            total += y.size(0)
            _, pred = torch.max(y_pred, 1)
            correct += (pred == y).sum().item()

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
        idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

        if opt.dev_file:
            p, r, f = norm_utils.evaluate(dev_data, dictionary, dictionary_reverse, vsm_model, None, None, d, isMeddra_dict)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            if fold_idx is None:
                torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
            else:
                torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx+1)))

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

    if len(opt.dev_file) == 0:
        torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))

    return best_dev_p, best_dev_r, best_dev_f





