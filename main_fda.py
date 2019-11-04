import random
import numpy as np
import torch
import os
import logging

from options import opt
import data
import train
from my_utils import makedir_and_clear
import test_fda
from test_fda import load_meddra_dict
import vsm
import multi_sieve
import norm_neural
import ensemble


logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.info(opt)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)



if opt.whattodo == 1:
    d = data.Data(opt)
    logging.info(d.config)

    makedir_and_clear(opt.output)

    if opt.cross_validation > 1:

        documents = data.load_data_fda(opt.train_file, True, opt.types, opt.type_filter, True, False)

        external_train_data = []
        if 'ext_corpus' in d.config:
            ext_corpus = d.config['ext_corpus']
            for k,v in ext_corpus.items():
                if k == 'made' or k == 'cardio':
                    external_train_data.extend(data.loadData(v['path'], True, v.get('types'), v.get('types')))
                elif k == 'tac':
                    external_train_data.extend(data.load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, False))
                else:
                    raise RuntimeError("not support external corpus")

        logging.info("use {} fold cross validataion".format(opt.cross_validation))
        fold_num = opt.cross_validation
        total_doc_num = len(documents)
        dev_doc_num = total_doc_num // fold_num

        macro_p = 0.0
        macro_r = 0.0
        macro_f = 0.0

        for fold_idx in range(fold_num):

            fold_start = fold_idx*dev_doc_num
            fold_end = fold_idx*dev_doc_num+dev_doc_num
            if fold_end > total_doc_num:
                fold_end = total_doc_num
            if fold_idx == fold_num-1 and fold_end < total_doc_num:
                fold_end = total_doc_num


            d.train_data = []
            d.train_data.extend(documents[:fold_start])
            d.train_data.extend(documents[fold_end:])
            if len(external_train_data) != 0:
                d.train_data.extend(external_train_data)
            d.dev_data = documents[fold_start:fold_end]

            logging.info("begin fold {}".format(fold_idx))

            logging.info("build alphabet ...")
            d.build_alphabet(d.train_data)
            d.build_alphabet(d.dev_data)
            d.fix_alphabet()

            logging.info("generate instance ...")
            d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet,
                                                            d.label_alphabet, d)
            d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet, d)

            logging.info("load pretrained word embedding ...")
            d.pretrain_word_embedding, d.word_emb_dim = data.build_pretrain_embedding(opt.word_emb_file, d.word_alphabet,
                                                                                      opt.word_emb_dim, False)

            p, r, f = train.train(d, opt, fold_idx)

            d.clear()
            d.save(os.path.join(opt.output, "data_{}.pkl".format(fold_idx+1)))

            macro_p += p
            macro_r += r
            macro_f += f


        logging.info("the macro averaged p r f are %.4f, %.4f, %.4f" % (macro_p*1.0/fold_num, macro_r*1.0/fold_num, macro_f*1.0/fold_num))

    else:
        # if -dev_file is assigned, we can use it for debugging
        # if not assign, we can train a model on the training set, but the model will be saved after final iteration.
        d.train_data = data.load_data_fda(opt.train_file, True, opt.types, opt.type_filter, True, False)

        external_train_data = []
        if 'ext_corpus' in d.config:
            ext_corpus = d.config['ext_corpus']
            for k,v in ext_corpus.items():
                if k == 'made' or k == 'cardio':
                    external_train_data.extend(data.loadData(v['path'], True, v.get('types'), v.get('types')))
                elif k == 'tac':
                    external_train_data.extend(data.load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, False))
                else:
                    raise RuntimeError("not support external corpus")
        if len(external_train_data) != 0:
            d.train_data.extend(external_train_data)

        if opt.dev_file:
            d.dev_data = data.load_data_fda(opt.dev_file, True, opt.types, opt.type_filter, True, False)
        else:
            logging.info("no dev data, the model will be saved after training finish")

        if opt.test_file:
            d.test_data = data.load_data_fda(opt.test_file, False, opt.types, opt.type_filter, True, False)



        logging.info("build alphabet ...")
        d.build_alphabet(d.train_data)
        if opt.dev_file:
            d.build_alphabet(d.dev_data)
        if opt.test_file:
            d.build_alphabet(d.test_data)
        d.fix_alphabet()
        logging.info("alphabet size {}".format(d.word_alphabet.size()))

        logging.info("generate instance ...")
        d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet,
                                                        d.label_alphabet, d)
        if opt.dev_file:
            d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet, d)

        logging.info("load pretrained word embedding ...")
        d.pretrain_word_embedding, d.word_emb_dim = data.build_pretrain_embedding(opt.word_emb_file, d.word_alphabet,
                                                                                  opt.word_emb_dim, False)

        p, r, f = train.train(d, opt, None)

        d.clear()
        d.save(os.path.join(opt.output, "data.pkl"))

elif opt.whattodo == 2:
    d = data.Data(opt)
    logging.info(d.config)

    makedir_and_clear(opt.output)

    if opt.cross_validation > 1:
        documents = data.load_data_fda(opt.train_file, True, opt.types, opt.type_filter, True, True)


        logging.info("use {} fold cross validataion".format(opt.cross_validation))
        fold_num = opt.cross_validation
        total_doc_num = len(documents)
        dev_doc_num = total_doc_num // fold_num

        macro_p = 0.0
        macro_r = 0.0
        macro_f = 0.0

        meddra_dict = load_meddra_dict(d)


        for fold_idx in range(fold_num):

            fold_start = fold_idx*dev_doc_num
            fold_end = fold_idx*dev_doc_num+dev_doc_num
            if fold_end > total_doc_num:
                fold_end = total_doc_num
            if fold_idx == fold_num-1 and fold_end < total_doc_num:
                fold_end = total_doc_num

            train_data = []
            train_data.extend(documents[:fold_start])
            train_data.extend(documents[fold_end:])
            dev_data = documents[fold_start:fold_end]

            logging.info("begin fold {}".format(fold_idx))
            logging.info("doc start {}, doc end {}".format(fold_start, fold_end))

            if opt.norm_rule and opt.norm_vsm and opt.norm_neural:  # ensemble
                p, r, f = ensemble.train(train_data, dev_data, d, meddra_dict, opt, fold_idx, True)
            elif opt.norm_rule:
                p,r,f = multi_sieve.train(train_data, dev_data, d, meddra_dict, opt, fold_idx, True)
            elif opt.norm_vsm:
                p, r, f = vsm.train(train_data, dev_data, None, d, meddra_dict, None, opt, fold_idx, True)
            elif opt.norm_neural:
                p, r, f = norm_neural.train(train_data, dev_data, None, d, meddra_dict, None, opt, fold_idx, True)
            else:
                raise RuntimeError("wrong configuration")

            macro_p += p
            macro_r += r
            macro_f += f


        logging.info("the macro averaged p r f are %.4f, %.4f, %.4f" % (macro_p*1.0/fold_num, macro_r*1.0/fold_num, macro_f*1.0/fold_num))



    else:
        train_data = data.load_data_fda(opt.train_file, True, opt.types, opt.type_filter, True, True)

        if opt.dev_file:
            dev_data = data.load_data_fda(opt.dev_file, True, opt.types, opt.type_filter, True, True)
        else:
            logging.info("no dev data, the model will be saved after training finish")
            dev_data = None

        if opt.test_file:
            test_data = data.load_data_fda(opt.test_file, False, opt.types, opt.type_filter, True, True)
        else:
            test_data = None

        meddra_dict = load_meddra_dict(d)


        if opt.norm_rule and opt.norm_vsm and opt.norm_neural:  # ensemble
            ensemble.train(train_data, dev_data, test_data, d, meddra_dict, None, opt, None, True)
        elif opt.norm_vsm:
            vsm.train(train_data, dev_data, test_data, d, meddra_dict, None, opt, None, True)
        elif opt.norm_neural:
            norm_neural.train(train_data, dev_data, test_data, d, meddra_dict, None, opt, None, True)
        else:
            raise RuntimeError("wrong configuration")




else:

    d = data.Data(opt)
    d.load(os.path.join(opt.output, "data.pkl"))

    d_new = data.Data(opt)
    d.config = d_new.config
    logging.info(d.config)

    test_fda.test(d, opt)

