import nltk
import os
from seqmodel import SeqModel
import torch
from my_utils import makedir_and_clear, evaluate, is_overlapped
import time
from data import processOneFile_fda, read_instance_from_one_document
import logging
from metric import get_ner_BIOHD_1234
from data_structure import Entity
import xml.dom
import codecs
import multi_sieve
import re
import vsm
import norm_neural
import copy
import ensemble

def span_to_start(entity):
    ret = ""
    for span in entity.spans:
        ret += str(span[0])+","
    return ret[:-1]

def span_to_len(entity):
    ret = ""
    for span in entity.spans:
        ret += str(span[1]-span[0])+","
    return ret[:-1]

def dump_results(doc_name, entities, opt, annotation_file):
    entity_id = 1
    dom1 = xml.dom.getDOMImplementation()
    doc = dom1.createDocument(None, "SubmissionLabel", None)
    xml_SubmissionLabel = doc.documentElement
    xml_SubmissionLabel.setAttribute('drug', doc_name[:doc_name.find(".xml")])

    xml_Text = doc.createElement('Text')
    xml_SubmissionLabel.appendChild(xml_Text)

    for section in annotation_file.sections:
        xml_Section = doc.createElement('Section')
        xml_Section.setAttribute('id', section.id)
        xml_Section.setAttribute('name', section.name)
        xml_Section_text = doc.createTextNode(section.text)
        xml_Section.appendChild(xml_Section_text)
        xml_Text.appendChild(xml_Section)

    xml_IgnoredRegions = doc.createElement('IgnoredRegions')
    xml_SubmissionLabel.appendChild(xml_IgnoredRegions)

    for ignore_region in annotation_file.ignore_regions:
        xml_IgnoredRegion = doc.createElement('IgnoredRegion')
        xml_IgnoredRegion.setAttribute('len', str(ignore_region.end-ignore_region.start))
        xml_IgnoredRegion.setAttribute('name', ignore_region.name)
        xml_IgnoredRegion.setAttribute('section', ignore_region.section)
        xml_IgnoredRegion.setAttribute('start', str(ignore_region.start))
        xml_IgnoredRegions.appendChild(xml_IgnoredRegion)

    xml_Mentions = doc.createElement('Mentions')
    xml_SubmissionLabel.appendChild(xml_Mentions)

    for entity in entities:
        xml_Mention = doc.createElement('Mention')
        xml_Mention.setAttribute('id', 'M'+str(entity_id))
        entity_id += 1
        xml_Mention.setAttribute('len', span_to_len(entity))
        xml_Mention.setAttribute('section', entity.section)
        xml_Mention.setAttribute('start', span_to_start(entity))
        xml_Mention.setAttribute('type', 'OSE_Labeled_AE')

        if len(entity.norm_ids) == 0:
            xml_Normalization = doc.createElement('Normalization')
            xml_Normalization.setAttribute('meddra_pt', '')
            xml_Normalization.setAttribute('meddra_pt_id', '')
            xml_Mention.appendChild(xml_Normalization)
        else:
            for idx, norm_id in enumerate(entity.norm_ids):
                norm_name = entity.norm_names[idx]
                xml_Normalization = doc.createElement('Normalization')
                xml_Normalization.setAttribute('meddra_pt', norm_name)
                xml_Normalization.setAttribute('meddra_pt_id', norm_id)
                xml_Mention.appendChild(xml_Normalization)

        xml_Mentions.appendChild(xml_Mention)


    with codecs.open(os.path.join(opt.predict, doc_name), 'w', 'UTF-8') as fp:
        doc.writexml(fp, addindent=' ' * 2, newl='\n', encoding='UTF-8')



def remove_entity_in_the_ignore_region(ignore_regions, entities, section_id):
    ret = []

    for entity in entities:
        remove_this_entity = False
        for ignore_region in ignore_regions:
            if ignore_region.section != section_id:
                continue
            for span in entity.spans:
                if is_overlapped(span[0], span[1], ignore_region.start, ignore_region.end):
                    remove_this_entity = True
                    break
            if remove_this_entity:
                break
        if not remove_this_entity:
            entity.section = section_id
            ret.append(entity)

    return ret

def translateResultsintoEntities(sentences, predict_results):
    pred_entities = []
    sent_num = len(predict_results)
    for idx in range(sent_num):

        predict_list = predict_results[idx]
        sentence = sentences[idx]

        entities = get_ner_BIOHD_1234(predict_list, False)

        # find span based on tkSpan, fill name
        for entity in entities:
            name = ''
            for tkSpan in entity.tkSpans:
                span = [sentence[tkSpan[0]]['start'], sentence[tkSpan[1]]['end']]
                entity.spans.append(span)
                for i in range(tkSpan[0], tkSpan[1]+1):
                    name += sentence[i]['text'] + ' '
            entity.name = name.strip()

        pred_entities.extend(entities)


    return pred_entities

def test(data, opt):

    corpus_dir = opt.test_file

    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise RuntimeError("invalid nlp tool")

    corpus_files = [f for f in os.listdir(corpus_dir) if f.find('.xml') != -1]

    model = SeqModel(data, opt)
    if opt.test_in_cpu:
        model.load_state_dict(
            torch.load(os.path.join(opt.output, 'model.pkl'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.output, 'model.pkl')))

    meddra_dict = load_meddra_dict(data)

    # initialize norm models
    if opt.norm_rule and opt.norm_vsm and opt.norm_neural: # ensemble
        logging.info("use ensemble normer")
        multi_sieve.init(opt, None, data, meddra_dict, None, True)
        if opt.ensemble == 'learn':
            if opt.test_in_cpu:
                ensemble_model = torch.load(os.path.join(opt.output, 'ensemble.pkl'), map_location='cpu')
            else:
                ensemble_model = torch.load(os.path.join(opt.output, 'ensemble.pkl'))
            ensemble_model.eval()
        else:
            if opt.test_in_cpu:
                vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
                neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
            else:
                vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
                neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))

            vsm_model.eval()
            neural_model.eval()

    elif opt.norm_rule:
        logging.info("use rule-based normer")
        multi_sieve.init(opt, None, data, meddra_dict)

    elif opt.norm_vsm:
        logging.info("use vsm-based normer")
        if opt.test_in_cpu:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'), map_location='cpu')
        else:
            vsm_model = torch.load(os.path.join(opt.output, 'vsm.pkl'))
        vsm_model.eval()

    elif opt.norm_neural:
        logging.info("use neural-based normer")
        if opt.test_in_cpu:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'), map_location='cpu')
        else:
            neural_model = torch.load(os.path.join(opt.output, 'norm_neural.pkl'))
        neural_model.eval()
    else:
        logging.info("no normalization is performed.")


    makedir_and_clear(opt.predict)

    ct_success = 0
    ct_error = 0

    for fileName in corpus_files:
        try:
            start = time.time()
            document, annotation_file = processOneFile_fda(fileName, corpus_dir, nlp_tool, False, opt.types, opt.type_filter, True, False)
            pred_entities = []

            for section in document:

                data.test_texts = []
                data.test_Ids = []
                read_instance_from_one_document(section, data.word_alphabet, data.char_alphabet, data.label_alphabet,
                                                data.test_texts, data.test_Ids, data)

                _, _, _, _, _, pred_results, _ = evaluate(data, opt, model, 'test', False, opt.nbest)

                entities = translateResultsintoEntities(section.sentences, pred_results)

                # remove the entity in the ignore_region and fill section_id
                section_id = section.name[section.name.rfind('_')+1: ]
                entities = remove_entity_in_the_ignore_region(annotation_file.ignore_regions, entities, section_id)


                if opt.norm_rule and opt.norm_vsm and opt.norm_neural:
                    if opt.ensemble == 'learn':
                        ensemble_model.process_one_doc(section, entities, meddra_dict, None, True)
                    else:
                        pred_entities1 = copy.deepcopy(entities)
                        pred_entities2 = copy.deepcopy(entities)
                        pred_entities3 = copy.deepcopy(entities)
                        multi_sieve.runMultiPassSieve(section, pred_entities1, meddra_dict, True)
                        vsm_model.process_one_doc(section, pred_entities2, meddra_dict, None, True)
                        neural_model.process_one_doc(section, pred_entities3, meddra_dict, None, True)

                        # merge pred_entities1, pred_entities2, pred_entities3 into entities
                        ensemble.merge_result(pred_entities1, pred_entities2, pred_entities3, entities, meddra_dict, True, vsm_model.dict_alphabet, data)

                elif opt.norm_rule:
                    multi_sieve.runMultiPassSieve(section, entities, meddra_dict, True)
                elif opt.norm_vsm:
                    vsm_model.process_one_doc(section, entities, meddra_dict, None, True)
                elif opt.norm_neural:
                    neural_model.process_one_doc(section, entities, meddra_dict, None, True)


                for entity in entities:
                    if len(entity.norm_ids)!=0: # if a mention can't be normed, not output it
                        pred_entities.append(entity)


            dump_results(fileName, pred_entities, opt, annotation_file)

            end = time.time()
            logging.info("process %s complete with %.2fs" % (fileName, end - start))

            ct_success += 1
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            ct_error += 1

    if opt.norm_rule:
        multi_sieve.finalize(True)

    logging.info("test finished, total {}, error {}".format(ct_success + ct_error, ct_error))


def load_meddra_dict(data):
    input_path = data.config['norm_dict']

    map_id_to_name = dict()

    with codecs.open(input_path, 'r', 'UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line == u'':
                continue
            token = re.split(r"\|\|", line)
            cui = token[0]

            conceptNames = token[1]

            map_id_to_name[cui] = conceptNames


    return map_id_to_name