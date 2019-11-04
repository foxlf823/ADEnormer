import codecs
from alphabet import Alphabet
import numpy as np
import pickle as pk
from os import listdir
from os.path import isfile, join
from my_utils import get_bioc_file, get_text_file, normalize_word, is_overlapped
# import spacy
from data_structure import Entity, Document
from options import opt
import logging
import re
import nltk
# from my_corenlp_wrapper import StanfordCoreNLP
import json
import xml.sax
import fda_xml_handler

def getLabel(start, end, entities):
    match = ""
    for entity in entities:
        if start == entity.spans[0][0] and end == entity.spans[0][1] : # S
            match = "S"
            break
        elif start == entity.spans[0][0] and end != entity.spans[0][1] : # B
            match = "B"
            break
        elif start != entity.spans[0][0] and end == entity.spans[0][1] : # E
            match = "E"
            break
        elif start > entity.spans[0][0] and end < entity.spans[0][1]:  # M
            match = "M"
            break

    if match != "":
        if opt.no_type:
            return match + "-" +"X"
        else:
            return match+"-"+entity.type
    else:
        return "O"

def getLabel_BIOHD1234(sent, tokenIdx, entities, ignore_regions, section_id):
    # if token occur in ignored regions, its label should be 'O'

    if ignore_regions is not None:
        for ignore_region in ignore_regions:
            if ignore_region.section != section_id:
                continue
            if sent[tokenIdx][1] >= ignore_region.start and sent[tokenIdx][2] <= ignore_region.end:
                return 'O'

    # count the number that tok occurs in spans

    spansContainTok = [] # [[start, end]]
    entityContainSpan = []
    for entity in entities:
        for span in entity.spans:
            if sent[tokenIdx][1] >= span[0] and sent[tokenIdx][2] <= span[1]:
                spansContainTok.append(span)
                entityContainSpan.append(entity)

    if len(spansContainTok) == 0:
        return 'O'
    elif len(spansContainTok) > 1: # (DB DI {HB HI) DB DI}
        for span in spansContainTok:
            if sent[tokenIdx][1] == span[0]:
                return 'HB-X'
            else :
                return 'HI-X'
    else:
        currentSpan = spansContainTok[0]
        currentEntity = entityContainSpan[0]
        overlapped = False
        overlappedSpan = None

        for entity in entities:
            for span in entity.spans:
                if currentSpan[0] == span[0] and currentSpan[1] == span[1]:
                    continue

                # if (currentSpan[1] >= span[0] and currentSpan[0] <= span[1]) :
                if is_overlapped(currentSpan[0], currentSpan[1], span[0], span[1]):
                    overlapped = True
                    overlappedSpan = span
                    break

        if overlapped: # (DB DI {HB HI) DB DI}
            i = 0
            while i<len(sent):
                if (sent[i][1] >= overlappedSpan[1]):
                    break
                i += 1

            if (sent[tokenIdx][1] == currentSpan[0]):
                return 'D3B-X'
            elif i == tokenIdx:
                return 'D1B-X'
            else:
                if sent[i][1] < overlappedSpan[0]:
                    return 'D3I-X'
                else:
                    return 'D1I-X'
        else:
            if len(currentEntity.spans) > 1: # (DB DI)  (DB DI)
                otherSpan = None
                for span in currentEntity.spans:
                    if (currentSpan[0] == span[0] and currentSpan[1] == span[1]):
                        continue

                    otherSpan = span # assume only one other discontinuous span
                    break

                coutOtherSpan = 0
                for entity in entities:
                    for span in entity.spans:
                        # if (otherSpan[0] == span[0] and otherSpan[1] == span[1]):
                        if otherSpan[0] >= span[0] and otherSpan[1] <= span[1]:
                            coutOtherSpan += 1

                if (coutOtherSpan > 1):
                    if (currentSpan[0] < otherSpan[0]):
                        if (sent[tokenIdx][1] == currentSpan[0]):
                            return 'D3B-X'
                        else:
                            return 'D3I-X'
                    else:
                        if (sent[tokenIdx][1] == currentSpan[0]):
                            return 'D1B-X'
                        else:
                            return 'D1I-X'
                else:
                    if currentSpan[0] < otherSpan[0]:
                        if (sent[tokenIdx][1] == currentSpan[0]):
                            return 'D4B-X'
                        else:
                            return 'D4I-X'
                    else:
                        if (sent[tokenIdx][1] == currentSpan[0]):
                            return 'D2B-X'
                        else:
                            return 'D2I-X'

            else:
                if (sent[tokenIdx][1] == currentSpan[0]) : # (B I)
                    return 'B-X'
                else:
                    return 'I-X'




def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp, entities):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
            # if entities:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

# if add pos, add to the end, so external functions don't need to be modified too much
# def text_tokenize_and_postagging(txt, sent_start):
#     tokens= my_tokenize(txt)
#     pos_tags = nltk.pos_tag(tokens)
#
#     offset = 0
#     for token, pos_tag in pos_tags:
#         offset = txt.find(token, offset)
#         yield token, pos_tag, offset+sent_start, offset+len(token)+sent_start
#         offset += len(token)

def text_tokenize_and_postagging(txt, sent_start):
    tokens= my_tokenize(txt)
    pos_tags = nltk.pos_tag(tokens)

    offset = 0
    for token, pos_tag in pos_tags:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start, pos_tag
        offset += len(token)

def token_from_sent(txt, sent_start):
    return [token for token in text_tokenize_and_postagging(txt, sent_start)]

def get_sentences_and_tokens_from_nltk(text, nlp_tool, entities, ignore_regions, section_id):
    all_sents_inds = []
    generator = nlp_tool.span_tokenize(text)
    for t in generator:
        all_sents_inds.append(t)

    sentences = []
    for ind in range(len(all_sents_inds)):
        t_start = all_sents_inds[ind][0]
        t_end = all_sents_inds[ind][1]

        tmp_tokens = token_from_sent(text[t_start:t_end], t_start)
        sentence_tokens = []
        for token_idx, token in enumerate(tmp_tokens):
            token_dict = {}
            token_dict['start'], token_dict['end'] = token[1], token[2]
            token_dict['text'] = token[0]
            token_dict['pos'] = token[3]
            token_dict['cap'] = featureCapital(token[0])
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
                if opt.schema == 'BMES':
                    token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)
                elif opt.schema == 'BIOHD_1234':
                    token_dict['label'] = getLabel_BIOHD1234(tmp_tokens, token_idx, entities, ignore_regions, section_id)
                else:
                    raise RuntimeError("invalid label schema")

            sentence_tokens.append(token_dict)

        # debug feili
        # has_HBorDB = False
        # for token_dict in sentence_tokens:
        #     if token_dict['label'] in set(['HB-X', 'D1B-X', 'D2B-X', 'D3B-X', 'D4B-X']):
        #         has_HBorDB = True
        #         break
        # if has_HBorDB:
        #     sentences.append(sentence_tokens)
        sentences.append(sentence_tokens)
    return sentences

def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    text = text.encode("utf-8")
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    # if type(output) is str:
    if type(output) is unicode:
        output = json.loads(output, strict=False)
    return output

def get_sentences_and_tokens_from_stanford(text, nlp_tool, entities):
    stanford_output = get_stanford_annotations(text, nlp_tool)
    sentences = []
    temp = stanford_output['sentences']
    for sentence in stanford_output['sentences']:
        sentence_tokens = []
        for stanford_token in sentence['tokens']:
            token_dict = {}
            token_dict['start'] = int(stanford_token['characterOffsetBegin'])
            token_dict['end'] = int(stanford_token['characterOffsetEnd'])
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                                     token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences


def processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool, isTraining, types, type_filter):
    document = Document()
    document.name = fileName[:fileName.find('.')]


    ct_snomed = 0
    ct_meddra = 0
    ct_unnormed = 0

    if annotation_dir:
        annotation_file = get_bioc_file(join(annotation_dir, fileName))
        bioc_passage = annotation_file[0].passages[0]
        entities = []

        for entity in bioc_passage.annotations:
            if types and (entity.infons['type'] not in type_filter):
                continue
            entity_ = Entity()
            entity_.id = entity.id
            processed_name = entity.text.replace('\\n', ' ')
            if len(processed_name) == 0:
                logging.debug("{}: entity {} name is empty".format(fileName, entity.id))
                continue
            entity_.name = processed_name

            entity_.type = entity.infons['type']
            entity_.spans.append([entity.locations[0].offset,entity.locations[0].end])
            if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A')\
                    and ('SNOMED term' in entity.infons and entity.infons['SNOMED term'] != 'N/A'):
                entity_.norm_ids.append(entity.infons['SNOMED code'])
                entity_.norm_names.append(entity.infons['SNOMED term'])
                ct_snomed += 1
            elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A')\
                    and ('MedDRA term' in entity.infons and entity.infons['MedDRA term'] != 'N/A'):
                entity_.norm_ids.append(entity.infons['MedDRA code'])
                entity_.norm_names.append(entity.infons['MedDRA term'])
                ct_meddra += 1
            else:
                logging.debug("{}: no norm id in entity {}".format(fileName, entity.id))
                ct_unnormed += 1
                continue

            entities.append(entity_)

        document.entities = entities

    corpus_file = get_text_file(join(corpus_dir, fileName.split('.bioc')[0]))
    document.text = corpus_file

    if opt.nlp_tool == "spacy":
        if isTraining:
            sentences = get_sentences_and_tokens_from_spacy(corpus_file, nlp_tool, document.entities)
        else:
            sentences = get_sentences_and_tokens_from_spacy(corpus_file, nlp_tool, None)
    elif opt.nlp_tool == "nltk":
        if isTraining:
            sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, document.entities, None, None)
        else:
            sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, None, None, None)
    elif opt.nlp_tool == "stanford":
        if isTraining:
            sentences = get_sentences_and_tokens_from_stanford(corpus_file, nlp_tool, document.entities)
        else:
            sentences = get_sentences_and_tokens_from_stanford(corpus_file, nlp_tool, None)
    else:
        raise RuntimeError("invalid nlp tool")


    document.sentences = sentences

    return document, ct_snomed, ct_meddra, ct_unnormed

def get_fda_file(file_path):
    handler = fda_xml_handler.FdaXmlHandler()
    xml.sax.parse(file_path, handler)
    return handler

# for fda 2018 data, the entity is extractly the mentions in the document.
# for tac 2017 data, the gold normalizations can be mapped to mentions.
# so if isFDA2018-False and isNorm-True, the 'document.entities' will be fake mentions.
# otherwise, 'document.entities' are true mentions.
def processOneFile_fda(fileName, annotation_dir, nlp_tool, isTraining, types, type_filter, isFDA2018, isNorm):
    documents = []
    annotation_file = get_fda_file(join(annotation_dir, fileName))

    # each section is a document
    for section in annotation_file.sections:
        document = Document()
        document.name = fileName[:fileName.find('.')]+"_"+section.id
        if section.text is None:
            document.text = ""
            document.entities = []
            document.sentences = []
            documents.append(document)
            continue

        document.text = section.text

        entities = []

        if isFDA2018==False and isNorm==True:
            for reaction in annotation_file.reactions:
                entity = Entity()
                entity.name = reaction.name
                for normalization in reaction.normalizations:
                    entity.norm_ids.append(normalization.meddra_pt_id) # can be none
                    entity.norm_names.append(normalization.meddra_pt)
                entities.append(entity)

        else:
            for entity in annotation_file.mentions:
                if entity.section != section.id:
                    continue
                if types and (entity.type not in type_filter):
                    continue
                entities.append(entity)

        document.entities = entities

        if opt.nlp_tool == "nltk":
            if isTraining:
                sentences = get_sentences_and_tokens_from_nltk(section.text, nlp_tool, document.entities, annotation_file.ignore_regions, section.id)
            else:
                sentences = get_sentences_and_tokens_from_nltk(section.text, nlp_tool, None, annotation_file.ignore_regions, section.id)
        else:
            raise RuntimeError("invalid nlp tool")


        document.sentences = sentences

        documents.append(document)

    return documents, annotation_file



def loadData(basedir, isTraining, types, type_filter):

    logging.info("loadData: {}".format(basedir))

    list_dir = listdir(basedir)
    if 'bioc' in list_dir:
        annotation_dir = join(basedir, 'bioc')
    elif 'annotations' in list_dir:
        annotation_dir = join(basedir, 'annotations')
    else:
        raise RuntimeError("no bioc or annotations in {}".format(basedir))

    if 'txt' in list_dir:
        corpus_dir = join(basedir, 'txt')
    elif 'corpus' in list_dir:
        corpus_dir = join(basedir, 'corpus')
    else:
        raise RuntimeError("no txt or corpus in {}".format(basedir))

    # spacy, nltk, stanford
    if opt.nlp_tool == "spacy":
        nlp_tool = spacy.load('en')
    elif opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    elif opt.nlp_tool == "stanford":
        nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise RuntimeError("invalid nlp tool")

    documents = []

    count_document = 0
    count_sentence = 0
    count_entity = 0
    count_entity_snomed = 0
    count_entity_meddra = 0
    count_entity_without_normed = 0

    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]
    for fileName in annotation_files:
        try:
            document, p1, p2, p3 = processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool, isTraining, types, type_filter)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.append(document)

        # statistics
        count_document += 1
        count_sentence += len(document.sentences)
        count_entity += len(document.entities)
        count_entity_snomed += p1
        count_entity_meddra += p2
        count_entity_without_normed += p3

    logging.info("document number: {}".format(count_document))
    logging.info("sentence number: {}".format(count_sentence))
    logging.info("entity number {}, snomed {}, meddra {}, unnormed {}".format(count_entity, count_entity_snomed,
                                                                              count_entity_meddra, count_entity_without_normed))

    return documents

def load_data_fda(basedir, isTraining, types, type_filter, isFDA2018, isNorm):

    logging.info("load_data_fda: {}".format(basedir))

    # spacy, nltk, stanford
    if opt.nlp_tool == "spacy":
        nlp_tool = spacy.load('en')
    elif opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    elif opt.nlp_tool == "stanford":
        nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise RuntimeError("invalid nlp tool")

    documents = []

    count_document = 0
    count_section = 0
    count_sentence = 0
    count_entity = 0

    annotation_files = [f for f in listdir(basedir) if f.find('.xml')!=-1]
    for fileName in annotation_files:
        try:
            document, _ = processOneFile_fda(fileName, basedir, nlp_tool, isTraining, types, type_filter, isFDA2018, isNorm)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.extend(document)

        # statistics
        count_document += 1
        for d in document:
            count_section += 1
            count_sentence += len(d.sentences)
            count_entity += len(d.entities)

    logging.info("document number: {}".format(count_document))
    logging.info("section number: {}".format(count_section))
    logging.info("sentence number: {}".format(count_sentence))
    logging.info("entity number {}".format(count_entity))

    return documents

def read_instance_from_one_document(document, word_alphabet, char_alphabet, label_alphabet, instence_texts, instence_Ids, data_config):

    for sentence in document.sentences:

        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        label_Ids = []
        if data_config.feat_config is not None:
            features = []
            feature_Ids = []
        words_lower = []

        for token in sentence:
            word = token['text']
            if opt.ner_number_normalized:
                word = normalize_word(word)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
            if 'label' in token:
                labels.append(token['label'])
                label_Ids.append(label_alphabet.get_index(token['label']))

            if data_config.feat_config is not None:
                feat_list = []
                feat_Id = []
                for alphabet in data_config.feature_alphabets:
                    if alphabet.name == '[POS]':
                        feat_list.append(token['pos'])
                        feat_Id.append(alphabet.get_index(token['pos']))
                    elif alphabet.name == '[Cap]':
                        feat_list.append(token['cap'])
                        feat_Id.append(alphabet.get_index(token['cap']))
                features.append(feat_list)
                feature_Ids.append(feat_Id)

            # for elmo
            words_lower.append(word.lower())

            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        if len(labels) == 0:
            if data_config.feat_config is not None:
                instence_texts.append([words, chars, words_lower, features])
                instence_Ids.append([word_Ids, char_Ids, feature_Ids])
            else:
                instence_texts.append([words, chars, words_lower])
                instence_Ids.append([word_Ids, char_Ids])
        else:
            if data_config.feat_config is not None:
                instence_texts.append([words, chars, labels, words_lower, features])
                instence_Ids.append([word_Ids, char_Ids, label_Ids, feature_Ids])
            else:
                instence_texts.append([words, chars, labels, words_lower])
                instence_Ids.append([word_Ids, char_Ids, label_Ids])


def read_instance(data, word_alphabet, char_alphabet, label_alphabet, data_config):

    instence_texts = []
    instence_Ids = []

    for document in data:
        read_instance_from_one_document(document, word_alphabet, char_alphabet, label_alphabet, instence_texts,
                                        instence_Ids, data_config)

    return instence_texts, instence_Ids

# def _readString(f):
#     s = str()
#     c = f.read(1).decode('iso-8859-1')
#     while c != '\n' and c != ' ':
#         s = s + c
#         c = f.read(1).decode('iso-8859-1')
#
#     return s

def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break
                # try:
                #     embedd_dict[word.decode('utf-8')] = word_vector
                # except Exception , e:
                #     pass
                embedd_dict[word] = word_vector
    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
        # with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                # feili
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
                # embedd_dict[tokens[0].decode('utf-8')] = embedd

    return embedd_dict, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim, norm):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1
        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logging.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, dig_zero_match:%s, "
                 "case_dig_zero_match:%s, oov:%s, oov%%:%s"
                 %(pretrained_size, perfect_match, case_match, digits_replaced_with_zeros_found,
                   lowercase_and_digits_replaced_with_zeros_found, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


class Data:
    def __init__(self, opt):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)

        self.train_texts = None
        self.train_Ids = None
        self.dev_texts = None
        self.dev_Ids = None
        self.test_texts = None
        self.test_Ids = None

        self.pretrain_word_embedding = None
        self.word_emb_dim = opt.word_emb_dim

        self.config = self.read_config(opt.config)
        self.feat_config = None

        the_item = 'ner_feature'
        if the_item in self.config:
            self.feat_config = self.config[the_item] ## [POS]:{emb_size:20}
            self.feature_alphabets = []
            self.feature_emb_dims = []
            for k,v in self.feat_config.items():
                self.feature_alphabets.append(Alphabet(k))
                self.feature_emb_dims.append(int(v['emb_size']))



    def clear(self):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.train_texts = None
        self.train_Ids = None
        self.dev_texts = None
        self.dev_Ids = None
        self.test_texts = None
        self.test_Ids = None

        self.pretrain_word_embedding = None


    def build_alphabet(self, data):
        for document in data:
            for sentence in document.sentences:
                for token in sentence:
                    word = token['text']
                    if opt.ner_number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)
                    if token.get('label') is not None:
                        self.label_alphabet.add(token['label'])
                    # try:
                    #     self.label_alphabet.add(token['label'])
                    # except Exception, e:
                    #     print("document id {} {} {}".format(document.name))
                    #     exit()
                    if self.feat_config is not None:
                        for alphabet in self.feature_alphabets:
                            if alphabet.name == '[POS]':
                                alphabet.add(token['pos'])
                            elif alphabet.name == '[Cap]':
                                alphabet.add(token['cap'])

                    for char in word:
                        self.char_alphabet.add(char)


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pk.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pk.dump(self.__dict__, f, 2)
        f.close()

    def read_config(self, config_file):

        config = config_file_to_dict(config_file)
        return config

def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        line = line.strip()
        if line == '':
            continue
        if len(line) > 0 and line[0] == "#":
            continue

        pairs = line.split()
        if len(pairs) > 1:
            for idx, pair in enumerate(pairs):
                if idx == 0:
                    items = pair.split('=')
                    if items[0] not in config:
                        feat_dict = {}
                        config[items[0]] = feat_dict
                    feat_dict = config[items[0]]
                    feat_name = items[1]
                    one_dict = {}
                    feat_dict[feat_name] = one_dict
                else:
                    items = pair.split('=')
                    one_dict[items[0]] = items[1]
        else:
            items = pairs[0].split('=')
            if items[0] in config:
                print("Warning: duplicated config item found: %s, updated." % (items[0]))
            config[items[0]] = items[-1]

    return config

# def config_file_to_dict(input_file):
#     config = {}
#     fins = open(input_file, 'r').readlines()
#     for line in fins:
#         if len(line) > 0 and line[0] == "#":
#             continue
#         if "=" in line:
#             pair = line.strip().split('#', 1)[0].split('=', 1)
#             item = pair[0]
#             if item == "ner_feature":
#                 if item not in config:
#                     feat_dict = {}
#                     config[item] = feat_dict
#                 feat_dict = config[item]
#                 new_pair = pair[-1].split()
#                 feat_name = new_pair[0]
#                 one_dict = {}
#                 one_dict["emb_dir"] = None
#                 one_dict["emb_size"] = 10
#                 one_dict["emb_norm"] = False
#                 if len(new_pair) > 1:
#                     for idx in range(1, len(new_pair)):
#                         conf_pair = new_pair[idx].split('=')
#                         if conf_pair[0] == "emb_dir":
#                             one_dict["emb_dir"] = conf_pair[-1]
#                         elif conf_pair[0] == "emb_size":
#                             one_dict["emb_size"] = int(conf_pair[-1])
#                         elif conf_pair[0] == "emb_norm":
#                             one_dict["emb_norm"] = str2bool(conf_pair[-1])
#                 feat_dict[feat_name] = one_dict
#                 # print "feat",feat_dict
#             elif item == "ext_corpus":
#                 if item not in config:
#                     feat_dict = {}
#                     config[item] = feat_dict
#                 feat_dict = config[item]
#                 new_pair = pair[-1].split()
#                 feat_name = new_pair[0]
#                 one_dict = {}
#                 if len(new_pair) > 1:
#                     for idx in range(1, len(new_pair)):
#                         conf_pair = new_pair[idx].split('=')
#                         if conf_pair[0] == 'types':
#                             one_dict[conf_pair[0]] = set(conf_pair[1].split(','))
#                         else:
#                             one_dict[conf_pair[0]] = conf_pair[1]
#                 feat_dict[feat_name] = one_dict
#             else:
#                 if item in config:
#                     print("Warning: duplicated config item found: %s, updated." % (pair[0]))
#                 config[item] = pair[-1]
#     return config

def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False


def featureCapital(word):
    if word[0].isalpha() and word[0].isupper():
        return "1"
    else:
        return "0"




