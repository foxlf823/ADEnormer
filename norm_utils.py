
from options import opt
from my_utils import normalize_word
from data import my_tokenize
from data_structure import Entity
import multi_sieve
import copy
import logging
import ensemble


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def word_preprocess(word):
    if opt.norm_number_normalized:
        word = normalize_word(word)
    word = word.lower()
    return word


def build_alphabet(alphabet, data):
    for document in data:
        for sentence in document.sentences:
            for token in sentence:
                word = token['text']
                alphabet.add(word_preprocess(word))

def open_alphabet(alphabet):
    alphabet.open()

def fix_alphabet(alphabet):
    alphabet.close()

def build_alphabet_from_dict(alphabet, dictionary, isMeddra_dict):
    if isMeddra_dict:
        for concept_id, concept_name in dictionary.items():
            tokens = my_tokenize(concept_name)
            for word in tokens:
                alphabet.add(word_preprocess(word))
    else:
        for concept_id, concept in dictionary.items():
            for concept_name in concept.names:
                tokens = my_tokenize(concept_name)
                for word in tokens:
                    alphabet.add(word_preprocess(word))

def get_dict_index(dict_alphabet, concept_id):
    index = dict_alphabet.get_index(concept_id)-2 # since alphabet begin at 2
    return index

def get_dict_name(dict_alphabet, concept_index):
    name = dict_alphabet.get_instance(concept_index+2)
    return name

def init_dict_alphabet(dict_alphabet, dictionary):
    # concept_name may be string or umls_concept
    for concept_id, concept_name in dictionary.items():
        dict_alphabet.add(concept_id)

def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-2


def evaluate(documents, dictionary, dictionary_reverse, vsm_model, neural_model, ensemble_model, d, isMeddra_dict):
    if vsm_model is not None :
        vsm_model.eval()

    if neural_model is not None:
        neural_model.eval()

    if ensemble_model is not None:
        ensemble_model.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    # if opt.norm_rule and opt.norm_vsm and opt.norm_neural:
    #     ct_correct_rule = 0
    #     ct_correct_vsm = 0
    #     ct_correct_neural = 0
    #     ct_correct_all = 0
    #     ct_correct_rule_vsm = 0
    #     ct_correct_rule_neural = 0
    #     ct_correct_vsm_neural = 0

    for document in documents:

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

        if opt.norm_rule and opt.norm_vsm and opt.norm_neural:
            if opt.ensemble == 'learn':
                ensemble_model.process_one_doc(document, pred_entities, dictionary, dictionary_reverse, isMeddra_dict)
            else:
                pred_entities2 = copy.deepcopy(pred_entities)
                pred_entities3 = copy.deepcopy(pred_entities)
                merge_entities = copy.deepcopy(pred_entities)
                multi_sieve.runMultiPassSieve(document, pred_entities, dictionary, isMeddra_dict)
                vsm_model.process_one_doc(document, pred_entities2, dictionary, dictionary_reverse, isMeddra_dict)
                neural_model.process_one_doc(document, pred_entities3, dictionary, dictionary_reverse, isMeddra_dict)
        elif opt.norm_rule:
            multi_sieve.runMultiPassSieve(document, pred_entities, dictionary, isMeddra_dict)
        elif opt.norm_vsm:
            vsm_model.process_one_doc(document, pred_entities, dictionary, dictionary_reverse, isMeddra_dict)
        elif opt.norm_neural:
            neural_model.process_one_doc(document, pred_entities, dictionary, dictionary_reverse, isMeddra_dict)
        else:
            raise RuntimeError("wrong configuration")

        if opt.norm_rule and opt.norm_vsm and opt.norm_neural:

            # ct_gold += len(document.entities)
            # ct_predicted += len(pred_entities)
            # up bound of ensemble, if at least one system makes a correct prediction, we count it as correct.
            # for idx, gold in enumerate(document.entities):
                # if (pred_entities[idx].rule_id is not None and pred_entities[idx].rule_id in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is not None and pred_entities2[idx].vsm_id in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is not None and pred_entities3[idx].neural_id in gold.norm_ids):
                #     ct_correct_all += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is not None and pred_entities[idx].rule_id in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is None or pred_entities2[idx].vsm_id not in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is None or pred_entities3[idx].neural_id not in gold.norm_ids):
                #     ct_correct_rule += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is None or pred_entities[idx].rule_id not in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is not None and pred_entities2[idx].vsm_id in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is None or pred_entities3[idx].neural_id not in gold.norm_ids):
                #     ct_correct_vsm += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is None or pred_entities[idx].rule_id not in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is None or pred_entities2[idx].vsm_id not in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is not None and pred_entities3[idx].neural_id in gold.norm_ids):
                #     ct_correct_neural += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is not None and pred_entities[idx].rule_id in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is not None and pred_entities2[idx].vsm_id in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is None or pred_entities3[idx].neural_id not in gold.norm_ids):
                #     ct_correct_rule_vsm += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is not None and pred_entities[idx].rule_id in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is None or pred_entities2[idx].vsm_id not in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is not None and pred_entities3[idx].neural_id in gold.norm_ids):
                #     ct_correct_rule_neural += 1
                #     ct_correct += 1
                #
                # if (pred_entities[idx].rule_id is None or pred_entities[idx].rule_id not in gold.norm_ids)\
                #     and (pred_entities2[idx].vsm_id is not None and pred_entities2[idx].vsm_id in gold.norm_ids) \
                #         and (pred_entities3[idx].neural_id is not None and pred_entities3[idx].neural_id in gold.norm_ids):
                #     ct_correct_vsm_neural += 1
                #     ct_correct += 1

            if opt.ensemble == 'learn':

                if isMeddra_dict:
                    p1, p2, p3 = evaluate_for_fda(document.entities, pred_entities)
                else:
                    p1, p2, p3 = evaluate_for_ehr(document.entities, pred_entities, dictionary)

                ct_gold += p1
                ct_predicted += p2
                ct_correct += p3

            else:
                ensemble.merge_result(pred_entities, pred_entities2, pred_entities3, merge_entities, dictionary, isMeddra_dict, vsm_model.dict_alphabet, d)

                if isMeddra_dict:
                    p1, p2, p3 = evaluate_for_fda(document.entities, merge_entities)
                else:
                    p1, p2, p3 = evaluate_for_ehr(document.entities, merge_entities, dictionary)

                ct_gold += p1
                ct_predicted += p2
                ct_correct += p3


        else:

            if isMeddra_dict:
                p1, p2, p3 = evaluate_for_fda(document.entities, pred_entities)
            else:
                p1, p2, p3 = evaluate_for_ehr(document.entities, pred_entities, dictionary)

            ct_gold += p1
            ct_predicted += p2
            ct_correct += p3

    # if opt.norm_rule and opt.norm_vsm and opt.norm_neural:
    #     logging.info("ensemble correct. all:{} rule:{} vsm:{} neural:{} rule_vsm:{} rule_neural:{} vsm_neural:{}"
    #                  .format(ct_correct_all, ct_correct_rule, ct_correct_vsm, ct_correct_neural, ct_correct_rule_vsm,
    #                          ct_correct_rule_neural, ct_correct_vsm_neural))
    #
    # logging.info("gold:{} pred:{} correct:{}".format(ct_gold, ct_predicted, ct_correct))

    if ct_gold == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure


def evaluate_for_fda(gold_entities, pred_entities):

    ct_gold = len(gold_entities)
    ct_predicted = len(pred_entities)
    ct_correct = 0
    for idx, pred in enumerate(pred_entities):
        gold = gold_entities[idx]
        if len(pred.norm_ids) != 0 and pred.norm_ids[0] in gold.norm_ids:
            ct_correct += 1

    return ct_gold, ct_predicted, ct_correct

def evaluate_for_ehr(gold_entities, pred_entities, dictionary):

    ct_norm_gold = len(gold_entities)
    ct_norm_predict = len(pred_entities)
    ct_norm_correct = 0

    for predict_entity in pred_entities:

        for gold_entity in gold_entities:

            if predict_entity.equals_span(gold_entity):

                if len(gold_entity.norm_ids) == 0:
                    # if gold_entity not annotated, we count it as TP
                    ct_norm_correct += 1
                else:

                    if len(predict_entity.norm_ids) != 0 and predict_entity.norm_ids[0] in dictionary:
                        concept = dictionary[predict_entity.norm_ids[0]]

                        if gold_entity.norm_ids[0] in concept.codes:
                            ct_norm_correct += 1

                break

    return ct_norm_gold, ct_norm_predict, ct_norm_correct


