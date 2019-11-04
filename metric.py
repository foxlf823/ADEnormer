
import numpy as np
import math
import sys
import os
from data_structure import Entity



## input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list, True)
            pred_matrix = get_ner_BMES(predict_list, True)
        elif label_type == 'BIOHD_1234':
            gold_matrix = get_ner_BIOHD_1234(golden_list, True)
            pred_matrix = get_ner_BIOHD_1234(predict_list, True)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    # print "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


# def get_ner_BMES(label_list):
#
#     list_len = len(label_list)
#     begin_label = 'B-'
#     end_label = 'E-'
#     single_label = 'S-'
#     whole_tag = ''
#     index_tag = ''
#     tag_list = []
#     stand_matrix = []
#     for i in range(0, list_len):
#         # wordlabel = word_list[i]
#         current_label = label_list[i].upper()
#         if begin_label in current_label:
#             if index_tag != '':
#                 tag_list.append(whole_tag + ',' + str(i-1))
#             whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
#             index_tag = current_label.replace(begin_label,"",1)
#
#         elif single_label in current_label:
#             if index_tag != '':
#                 tag_list.append(whole_tag + ',' + str(i-1))
#             whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
#             tag_list.append(whole_tag)
#             whole_tag = ""
#             index_tag = ""
#         elif end_label in current_label:
#             if index_tag != '':
#                 tag_list.append(whole_tag +',' + str(i))
#             whole_tag = ''
#             index_tag = ''
#         else:
#             continue
#     if (whole_tag != '')&(index_tag != ''):
#         tag_list.append(whole_tag)
#     tag_list_len = len(tag_list)
#
#     for i in range(0, tag_list_len):
#         if  len(tag_list[i]) > 0:
#             tag_list[i] = tag_list[i]+ ']'
#             insert_list = reverse_style(tag_list[i])
#             stand_matrix.append(insert_list)
#
#     return stand_matrix

def checkWrongState_BMES(labelSequence, size):
    positionNew = -1
    positionOther = -1
    currentLabel = labelSequence[size - 1]
    assert currentLabel[0] == 'M' or currentLabel[0] == 'E'

    j = size - 2
    while j >= 0:
        if positionNew == -1 and currentLabel[2:] == labelSequence[j][2:] and labelSequence[j][0] == 'B' :
            positionNew = j
        elif positionOther == -1 and (currentLabel[2:] != labelSequence[j][2:] or labelSequence[j][0] != 'M'):
            positionOther = j

        if positionOther != -1 and positionNew != -1:
            break

        j -= 1

    if positionNew == -1:
        return False
    elif positionOther < positionNew:
        return True
    else:
        return False

def get_ner_BMES(outputs, return_str_or_not):
    entities = []

    for idx in range(len(outputs)):
        labelName = outputs[idx]

        if labelName[0] == 'S' or labelName[0] == 'B':
            entity = Entity()
            entity.type = labelName[2:]
            entity.tkSpans.append([idx, idx])
            entity.labelSpans.append([labelName])
            entities.append(entity)

        elif labelName[0] == 'M' or labelName[0] == 'E':
            if checkWrongState_BMES(outputs, idx+1):
                entity = entities[-1]
                entity.tkSpans[-1][1] = idx
                entity.labelSpans[-1].append(labelName)

    anwserEntities = entities

    if return_str_or_not:
        # transfer Entity class into its str representation
        strEntities = []
        for answer in anwserEntities:
            strEntity = answer.type
            for tkSpan in answer.tkSpans:
                strEntity += '['+str(tkSpan[0])+','+str(tkSpan[1])+']'
            strEntities.append(strEntity)
        return strEntities
    else:
        return anwserEntities

def combineTwoEntity(a, b):
    c = Entity()
    c.type = a.type

    if (a.tkSpans[0][0] < b.tkSpans[0][0]):
        if (a.tkSpans[0][1] + 1 == b.tkSpans[0][0]):
            c.tkSpans.append([a.tkSpans[0][0], b.tkSpans[0][1]]);
        else:
            c.tkSpans.append(a.tkSpans[0])
            c.tkSpans.append(b.tkSpans[0])
    else:
        if (b.tkSpans[0][1] + 1 == a.tkSpans[0][0]):
            c.tkSpans.append([b.tkSpans[0][0], a.tkSpans[0][1]])
        else:
            c.tkSpans.append(b.tkSpans[0])
            c.tkSpans.append(a.tkSpans[0])

    return c

def checkWrongState(labelSequence, size):
    positionNew = -1
    positionOther = -1

    currentLabel = labelSequence[size - 1]

    j = size - 2
    while j >= 0:

        if (currentLabel == 'I-X'):
            if (positionNew == -1 and labelSequence[j] == 'B-X'):
                positionNew = j
            elif positionOther==-1 and labelSequence[j]!= 'I-X' :
                positionOther = j
        elif (currentLabel == 'HI-X'):
            if (positionNew == -1 and labelSequence[j] == 'HB-X') :
                positionNew = j
            elif (positionOther == -1 and labelSequence[j] != 'HI-X'):
                positionOther = j
        elif (currentLabel == 'D1I-X') :
            if (positionNew == -1 and labelSequence[j] == 'D1B-X'):
                positionNew = j
            elif (positionOther == -1 and labelSequence[j] != 'D1I-X'):
                positionOther = j
        elif (currentLabel == 'D2I-X') :
            if (positionNew == -1 and labelSequence[j] == 'D2B-X'):
                positionNew = j
            elif (positionOther == -1 and labelSequence[j] != 'D2I-X'):
                positionOther = j
        elif (currentLabel == 'D3I-X') :
            if (positionNew == -1 and labelSequence[j] == 'D3B-X'):
                positionNew = j
            elif (positionOther == -1 and labelSequence[j] != 'D3I-X'):
                positionOther = j
        else:
            if (positionNew == -1 and labelSequence[j] == 'D4B-X'):
                positionNew = j
            elif (positionOther == -1 and labelSequence[j] != 'D4I-X'):
                positionOther = j

        if (positionOther != -1 and positionNew != -1):
            break

        j -= 1

    if (positionNew == -1):
        return False
    elif (positionOther < positionNew):
        return True
    else:
        return False

def get_ner_BIOHD_1234(outputs, return_str_or_not):
    entities = []
    for idx in range(len(outputs)):
        labelName = outputs[idx]

        if labelName == 'B-X' or labelName == 'HB-X' or labelName == 'D1B-X' or labelName == 'D2B-X' \
                or labelName == 'D3B-X' or labelName == 'D4B-X':
            entity = Entity()
            entity.type = 'X'
            entity.tkSpans.append([idx, idx])
            entity.labelSpans.append([labelName])
            entities.append(entity)
        elif labelName == 'I-X' or labelName == 'HI-X' or labelName == 'D1I-X' or labelName == 'D2I-X' \
					or labelName == 'D3I-X' or labelName == 'D4I-X':
            if checkWrongState(outputs, idx+1) :
                entity = entities[-1]
                entity.tkSpans[-1][1] = idx
                entity.labelSpans[-1].append(labelName)

    # post-processing to rebuild entities
    postEntities = []
    HB_HI = []
    D1B_D1I = []
    D2B_D2I = []
    D3B_D3I = []
    D4B_D4I = []

    for temp in entities:
        labelSpan = temp.labelSpans[0]

        if labelSpan[0] == 'HB-X':
            HB_HI.append(temp)
        elif labelSpan[0]=='D1B-X':
            D1B_D1I.append(temp)
        elif (labelSpan[0] == 'D2B-X') :
            D2B_D2I.append(temp)
        elif (labelSpan[0] == 'D3B-X') :
            D3B_D3I.append(temp)
        elif (labelSpan[0] == 'D4B-X') :
            D4B_D4I.append(temp)
        else:
            postEntities.append(temp)

    if len(HB_HI) != 0:
        for d1b in D1B_D1I:
            # combine with the nearest head entity at left
            target = None
            for hb in HB_HI:
                if (hb.tkSpans[0][0] < d1b.tkSpans[0][0]):
                    target = hb
                else:
                    break

            if target is None:
                pass
            else:
                combined = combineTwoEntity(d1b, target)
                postEntities.append(combined)
                if len(D1B_D1I) == 1:
                    postEntities.append(target)

        for d3b in D3B_D3I:
            # combine with the nearest head entity at right
            target = None
            for hb in reversed(HB_HI):
                if (hb.tkSpans[0][0] > d3b.tkSpans[0][0]):
                    target = hb
                else:
                    break

            if target is None:
                pass
            else:
                combined = combineTwoEntity(d3b, target)
                postEntities.append(combined)
                if len(D3B_D3I) == 1:
                    postEntities.append(target)

    else:
        for d2b in D2B_D2I:

            # combine with the nearest non-head entity at left
            target = None
            for db in D1B_D1I:
                if (db.tkSpans[0][0] < d2b.tkSpans[0][0]) :
                    target = db
                else :
                    break

            for db in D2B_D2I:
                if (db.tkSpans[0][0] < d2b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] < db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            for db in D3B_D3I:
                if (db.tkSpans[0][0] < d2b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] < db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            for db in D4B_D4I:
                if (db.tkSpans[0][0] < d2b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] < db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            if target is None:
                pass
            else:
                combined = combineTwoEntity(d2b, target)
                postEntities.append(combined)

        for d4b in D4B_D4I:

            # combine with the nearest non-head entity at right
            target = None
            for db in reversed(D1B_D1I):
                if (db.tkSpans[0][0] > d4b.tkSpans[0][0]) :
                    target = db
                else :
                    break

            for db in reversed(D2B_D2I):
                if (db.tkSpans[0][0] > d4b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] > db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            for db in reversed(D3B_D3I):
                if (db.tkSpans[0][0] > d4b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] > db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            for db in reversed(D4B_D4I):
                if (db.tkSpans[0][0] > d4b.tkSpans[0][0]):
                    if (target is not None and target.tkSpans[0][0] > db.tkSpans[0][0]):
                        target = db
                    else:
                        target = db
                else :
                    break

            if target is None:
                pass
            else:
                combined = combineTwoEntity(d4b, target)
                postEntities.append(combined)

    # resort by start position and remove the same entity
    anwserEntities = []
    for temp in postEntities:
        isIn = False
        for anwser in anwserEntities:
            if anwser.equalsTkSpan(temp):
                isIn = True
                break

        if isIn == False:
            iter = 0
            for old in anwserEntities:
                if old.tkSpans[0][0] > temp.tkSpans[0][0]:
                    break
                iter += 1

            anwserEntities.insert(iter, temp)

    if return_str_or_not:
        # transfer Entity class into its str representation
        strEntities = []
        for answer in anwserEntities:
            strEntity = 'X'
            for tkSpan in answer.tkSpans:
                strEntity += '['+str(tkSpan[0])+','+str(tkSpan[1])+']'
            strEntities.append(strEntity)
        return strEntities
    else:
        return anwserEntities


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels


def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])
            
    return sentences,golden_labels,predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file: {} {}".format(golden_file, predict_file))
    print("Label format: {}".format(label_type))
    golden_sent,golden_labels = readSentence(golden_file)
    predict_sent,predict_labels = readSentence(predict_file)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%sm R:%s, F:%s"%(P,R,F))



def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%s, R:%s, F:%s"%(P,R,F))



if __name__ == '__main__':
    # print "sys:",len(sys.argv)
    if len(sys.argv) == 3:
        fmeasure_from_singlefile(sys.argv[1],"BMES",int(sys.argv[2]))
    else:
        fmeasure_from_singlefile(sys.argv[1],"BMES")

