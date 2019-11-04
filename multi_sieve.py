# -*- coding: UTF-8 -*-
import os
import argparse
import codecs
import re
import shutil
from jpype import *
from data import get_fda_file, get_bioc_file
import logging
from data_structure import Entity
import norm_utils
from options import opt
import numpy as np

class Util:
    @classmethod
    def setMap(self, keyValueListMap, key, value):
        valueList = keyValueListMap.get(key)
        if valueList == None:
            valueList = list()
            keyValueListMap[key] = valueList
        valueList = Util.setList(valueList, value)
        return keyValueListMap

    @classmethod
    def setList(self, listt, value):
        if (value not in listt) and (value != u""):
            listt.append(value)
        return listt

    @classmethod
    def firstIndexOf(self, tokens, i, pattern):
        while i >=0:
            if re.match(pattern+r".*", tokens[i]):
                i -= 1
                return i
            i -= 1
        return -1

    @classmethod
    def read(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            return fp.read()

    @classmethod
    def containsAny(self, first, second):
        first_ = set(first)
        second_ = set(second)

        return len(first_ & second_) != 0

    @classmethod
    def getTokenIndex (self, tokens, token):
        i = 0
        while i < len(tokens):
            if tokens[i] == token:
                return i
            i += 1

        return -1

    @classmethod
    def addUnique(self, list, newList):
        for value in newList:
            list = Util.setList(list, value)
        return list




class Abbreviation:
    wikiAbbreviationExpansionListMap = dict()


    def __init__(self):
        self.textAbbreviationExpansionMap = dict()

    @classmethod
    def setWikiAbbreviationExpansionMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                token = re.split(r"\|\|", line)
                Abbreviation.wikiAbbreviationExpansionListMap = Util.setMap(Abbreviation.wikiAbbreviationExpansionListMap, token[0].lower(), token[1].lower())

    @classmethod
    def clearWikiAbbreviationExpansionMap(self):
        Abbreviation.wikiAbbreviationExpansionListMap.clear()

    @classmethod
    def getTentativeExpansion(self, tokens, i, abbreviationLength):
        expansion = u""
        while (i >= 0 and abbreviationLength > 0):
            expansion = tokens[i]+" "+expansion
            i -= 1
            abbreviationLength -= 1

        return expansion.strip()

    @classmethod
    def getExpansionByHearstAlgorithm(self, shortForm, longForm):
        sIndex = len(shortForm) - 1
        lIndex = len(longForm) - 1

        while(sIndex >= 0):
            currChar = shortForm[sIndex].lower()
            if not currChar.isalnum():
                sIndex -= 1
                continue

            while (((lIndex >= 0) and
                    (longForm[lIndex].lower() != currChar)) or
                    ((sIndex == 0) and (lIndex > 0) and
                    (longForm[lIndex-1].isalnum()))):
                lIndex -= 1

            if lIndex < 0:
                return u""

            lIndex -= 1
            sIndex -= 1

        lIndex = longForm.rfind(u" ", lIndex) + 1
        longForm = longForm[lIndex:]

        return longForm

    @classmethod
    def getEntireAbbreviation(self, text, string, indexes):
        if len(indexes) != 2:
            return string
        begin = int(indexes[0])
        end = int(indexes[1])
        if re.match(r"(^|\s|\W)[a-zA-Z]/"+string+r"/[a-zA-Z](\s|$|\W)", text[begin-3, end+3].lower()) :
            return text[begin-2, end+2].lower()
        elif re.matches(r"(^|\s|\W)"+string+r"/[a-zA-Z]/[a-zA-Z](\s|$|\W)", text[begin-1, end+5].lower()):
            return text[begin, end+4].lower()
        elif re.matches(r"(^|\s|\W)[a-zA-Z]/[a-zA-Z]/"+string+r"(\s|$|\W)", text[begin-5, end+1].lower()):
            return text[begin-4, end].lower()
        return string

    @classmethod
    def getBestExpansion(self, text, expansionList):
        maxNumberOfContentWords = 0
        maxContainedContentWords = 0
        returnExpansion = u""
        for expansion in expansionList:
            expansionContentWordsList = Ling.getContentWordsList(re.split(r"\s", expansion))
            tempNumberOfContentWords = len(expansionContentWordsList)
            tempContainedContentWords = 0
            for  expansionContentWord in expansionContentWordsList:
                if text.find(u" " + expansionContentWord) != -1 or text.find(expansionContentWord + u" ") != -1:
                    tempContainedContentWords += 1

            if tempNumberOfContentWords > maxNumberOfContentWords and tempContainedContentWords == tempNumberOfContentWords:
                maxNumberOfContentWords = tempNumberOfContentWords
                maxContainedContentWords = 1000
                returnExpansion = expansion
            elif tempNumberOfContentWords >= maxNumberOfContentWords and tempContainedContentWords > maxContainedContentWords:
                maxNumberOfContentWords = tempNumberOfContentWords
                maxContainedContentWords = tempContainedContentWords
                returnExpansion = expansion

        return returnExpansion

    @classmethod
    def getTrimmedExpansion(self, text, string, indexes, expansion):
        if len(indexes) != 2:
            return string
        begin = int(indexes[0])
        end = int(indexes[1])
        if re.matches(r"(^|\s|\W)[a-zA-Z]/"+string+r"/[a-zA-Z](\s|$|\W)", text[begin-3, end+3].lower()):
            return expansion[1].lower()
        elif re.matches(r"(^|\s|\W)"+string+r"/[a-zA-Z]/[a-zA-Z](\s|$|\W)", text[begin-1, end+5].lower()):
            return expansion[0].lower()
        elif re.matches(r"(^|\s|\W)[a-zA-Z]/[a-zA-Z]/"+string+r"(\s|$|\W)", text[begin-5, end+1].lower()):
            return expansion[2].lower()
        return string

    @classmethod
    def getAbbreviationExpansion(self, abbreviationObject, text, string, indexes):
        shortForm_longForm_map = abbreviationObject.getTextAbbreviationExpansionMap()
        stringTokens = re.split(r"\s", string)

        if len(stringTokens) == 1 and len(stringTokens[0]) == 1 :
            stringTokens[0] = Abbreviation.getEntireAbbreviation(text, string, re.split(r"\|", indexes))
        newString = u""

        for stringToken in stringTokens:
            if stringToken in shortForm_longForm_map:
                newString += shortForm_longForm_map.get(stringToken)+u" "
                continue
            candidateExpansionsList = Abbreviation.wikiAbbreviationExpansionListMap.get(stringToken) if stringToken in Abbreviation.wikiAbbreviationExpansionListMap else None

            if candidateExpansionsList == None:
                newString += stringToken + u" "
            else :
                expansion = candidateExpansionsList[0] if len(candidateExpansionsList) == 1 else Abbreviation.getBestExpansion(text, candidateExpansionsList)
                if expansion == u"":
                    newString += stringToken + u" "
                else:
                    newString += expansion + u" "

        if len(stringTokens) == 1 and stringTokens[0] != string:
            newString = getTrimmedExpansion(text, string, re.split(r"\|", indexes), re.split(r"/", newString))

        newString = newString.strip()
        return u"" if newString == (string) else newString


    def setTextAbbreviationExpansionMap_(self, tokens, abbreviationLength, abbreviation, expansionIndex):
        expansion = Abbreviation.getTentativeExpansion(tokens, expansionIndex, abbreviationLength)
        expansion = Abbreviation.getExpansionByHearstAlgorithm(abbreviation, expansion).lower().strip()
        if expansion != u"":
            self.textAbbreviationExpansionMap[abbreviation] = expansion


    def setTextAbbreviationExpansionMap (self, text):
        lines = re.split(r"\n+", text)
        for line in lines:
            line = line.strip()
            tokens = re.split(r"\s+", line)
            size = len(tokens)
            for i in range(size):
                expansionIndex = -1

                if (re.match(r"\(\w+(\-\w+)?\)(,|\.)?", tokens[i])) or (re.match(r"\([A-Z]+(;|,|\.)", tokens[i])):
                    expansionIndex = i - 1
                elif re.match(r"[A-Z]+\)", tokens[i]):
                    expansionIndex = Util.firstIndexOf(tokens, i, r"\(")

                if expansionIndex == -1:
                    continue

                abbreviation = tokens[i].replace(u"(", u"").replace(u")", u"").lower()
                reversedAbbreviation = Ling.reverse(abbreviation)

                if abbreviation[len(abbreviation) - 1] == u',' or abbreviation[len(abbreviation) - 1] == u'.' or abbreviation[len(abbreviation) - 1] == u';':
                    abbreviation = abbreviation[0: len(abbreviation) - 1]

                if (abbreviation in self.textAbbreviationExpansionMap) or (reversedAbbreviation in self.textAbbreviationExpansionMap):
                    continue

                abbreviationLength = len(abbreviation)
                self.setTextAbbreviationExpansionMap_(tokens, abbreviationLength, abbreviation, expansionIndex)
                if abbreviation not in self.textAbbreviationExpansionMap:
                    self.setTextAbbreviationExpansionMap_(tokens, abbreviationLength, reversedAbbreviation, expansionIndex)

    def getTextAbbreviationExpansionMap(self):
        return self.textAbbreviationExpansionMap

class Ling:
    stopwords = set()
    digitToWordMap = dict()
    wordToDigitMap = dict()
    suffixMap = dict()
    prefixMap = dict()
    affixMap = dict()
    logging.info("JVM class path {}".format(os.path.abspath(".")))
    startJVM(getDefaultJVMPath(), "-ea", "-Dfile.encoding=UTF-8", "-Djava.class.path={}".format(os.path.abspath(".")))
    PorterStemmer = JClass("PorterStemmer")
    AFFIX = u"ganglioma|cancer"
    PLURAL_DISORDER_SYNONYMS = [u"diseases", u"disorders", u"conditions", u"syndromes", u"symptoms",
                                             u"abnormalities", u"events", u"episodes", u"issues", u"impairments"]
    PREPOSITIONS = [u"in", u"with", u"on", u"of"]
    SINGULAR_DISORDER_SYNONYMS = [u"disease", u"disorder", u"condition", u"syndrome", u"symptom",
                                               u"abnormality", u"NOS", u"event", u"episode", u"issue", u"impairment"]

    def __init__(self):
        pass

    @classmethod
    def setStopwordsList(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if line == u'':
                    continue
                Ling.stopwords.add(line)

    @classmethod
    def getStopwordsList(self):
        return Ling.stopwords

    @classmethod
    def clearStopwordsList(self):
        Ling.stopwords.clear()

    @classmethod
    def setDigitToWordformMapAndReverse(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(r"\|\|", line)
                Ling.digitToWordMap = Util.setMap(Ling.digitToWordMap, tokens[0], tokens[1]);
                Ling.wordToDigitMap[tokens[1]]=tokens[0]

    @classmethod
    def clearDigitToWordformMapAndReverse(self):
        Ling.digitToWordMap.clear()
        Ling.wordToDigitMap.clear()

    @classmethod
    def setSuffixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(r"\|\|", line)
                if len(tokens) == 1:
                    values = Ling.suffixMap.get(tokens[0])
                    if values == None:
                        values = list()
                        Ling.suffixMap[tokens[0]]=values
                else:
                    Ling.suffixMap = Util.setMap(Ling.suffixMap, tokens[0], tokens[1])

    @classmethod
    def clearSuffixMap(self):
        Ling.suffixMap.clear()

    @classmethod
    def setPrefixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(r"\|\|", line)
                value = u"" if len(tokens) == 1 else tokens[1]
                Ling.prefixMap[tokens[0]] = value

    @classmethod
    def clearPrefixMap(self):
        Ling.prefixMap.clear()

    @classmethod
    def setAffixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(r"\|\|", line)
                value = u"" if len(tokens) == 1 else tokens[1]
                Ling.affixMap[tokens[0]] = value

    @classmethod
    def clearAffixMap(self):
        Ling.affixMap.clear()

    @classmethod
    def getStemmedPhrase(self, string):

        stemmed_name = u""
        str_tokens = re.split(r"\s+", string)
        for token in str_tokens:
            if token in Ling.stopwords:
                stemmed_name += token + u" "
                continue

            stemmed_token = Ling.PorterStemmer.get_stem(token).strip()
            if stemmed_token == u"":
                stemmed_token = token
            stemmed_name += stemmed_token + u" "

        stemmed_name = stemmed_name.strip()


        return stemmed_name

    @classmethod
    def reverse(self, string):
        reversedString = u""
        size = len(string)-1
        for i in range(size, -1, -1):
            reversedString += string[i]

        return reversedString

    @classmethod
    def getContentWordsList(self, words):
        contentWordsList = list()
        for word in words:
            if word in Ling.stopwords:
                continue
            contentWordsList = Util.setList(contentWordsList, word)

        return contentWordsList



    @classmethod
    def getStringPreposition(self, string):
        for preposition in Ling.PREPOSITIONS:
            if string.find(u" "+preposition+u" ") != -1:
                return preposition

        return u""

    @classmethod
    def getSubstring(self, tokens, begin, end):
        substring = u""
        i = begin
        while i < end:
            substring += tokens[i]+u" "
            i += 1

        substring = substring.strip()
        return substring

    @classmethod
    def getDigitToWordMap(self):

        return Ling.digitToWordMap

    @classmethod
    def getWordToDigitMap(self):
        return Ling.wordToDigitMap

    @classmethod
    def getSuffix_(self, str, len_):
        if len(str) < len_:
            return u""

        return str[len(str) - len_]

    @classmethod
    def getSuffix(self, str):
        if Ling.getSuffix_(str, 10) in Ling.suffixMap:
            return Ling.getSuffix_(str, 10)
        else:
            if Ling.getSuffix_(str, 7) in  Ling.suffixMap:
                return Ling.getSuffix_(str, 7)
            else:
                if Ling.getSuffix_(str, 6) in Ling.suffixMap:
                    return Ling.getSuffix_(str, 6)
                else:
                    if Ling.getSuffix_(str, 5) in Ling.suffixMap:
                        return Ling.getSuffix_(str, 5)
                    else:
                        if Ling.getSuffix_(str, 4) in Ling.suffixMap:
                            return Ling.getSuffix_(str, 4)
                        else:
                            if Ling.getSuffix_(str, 3) in Ling.suffixMap:
                                return Ling.getSuffix_(str, 3)
                            else:
                                if Ling.getSuffix_(str, 2) in Ling.suffixMap:
                                    return Ling.getSuffix_(str, 2)
                                else:
                                    return u""

    @classmethod
    def getSuffixMap(self):

        return Ling.suffixMap


    @classmethod
    def getPrefix_(self, str, len_):
        if len(str) < len_:
            return u""

        return str[0 : len_]

    @classmethod
    def getPrefix(self, str):
        if Ling.getPrefix_(str, 5) in Ling.prefixMap:
            return Ling.getPrefix_(str, 5)
        else:
            if Ling.getPrefix_(str, 4) in Ling.prefixMap:
                return Ling.getPrefix_(str, 4)
            else:
                if Ling.getPrefix_(str, 3) in Ling.prefixMap:
                    return Ling.getPrefix_(str, 3)
                else:
                    return u""

    @classmethod
    def getPrefixMap(self):
        return Ling.prefixMap

    @classmethod
    def getAffixMap(self):
        return Ling.affixMap


    @classmethod
    def getMatchingTokensCount(self, phrase1, phrase2):
        tokens = re.split(r"\s+", phrase1)

        temp = list()
        temp1 = re.split(r"\s+", phrase2)
        for t in tokens:
            if t in temp1:
                temp.append(t)
        tokens = temp

        temp = list()
        for t in tokens:
            if t in Ling.stopwords:
                continue
            temp.append(t)
        tokens = temp

        return 0 if len(tokens) == 0 else len(tokens)







class Terminology:

    def __init__(self):
        self.cuiAlternateCuiMap = dict()
        self.nameToCuiListMap = dict()
        self.cuiToNameListMap = dict()
        self.stemmedNameToCuiListMap = dict()
        self.cuiToStemmedNameListMap = dict()
        self.tokenToNameListMap = dict()
        self.compoundNameToCuiListMap = dict()
        self.simpleNameToCuiListMap = dict()

    def  getTokenToNameListMap(self):
        return self.tokenToNameListMap

    def  getSimpleNameToCuiListMap(self):
        return self.simpleNameToCuiListMap

    def getCuiToNameListMap(self):
        return self.cuiToNameListMap

    def getCompoundNameToCuiListMap(self):
        return self.compoundNameToCuiListMap


    def getStemmedNameToCuiListMap(self):
        return self.stemmedNameToCuiListMap

    def getNameToCuiListMap(self):

        return self.nameToCuiListMap


    def getCuiAlternateCuiMap(self):
        return self.cuiAlternateCuiMap


    def loadMaps(self, conceptName, cui):
        self.nameToCuiListMap = Util.setMap(self.nameToCuiListMap, conceptName, cui)
        self.cuiToNameListMap = Util.setMap(self.cuiToNameListMap, cui, conceptName)

        stemmedConceptName = Ling.getStemmedPhrase(conceptName)
        self.stemmedNameToCuiListMap = Util.setMap(self.stemmedNameToCuiListMap, stemmedConceptName, cui)
        self.cuiToStemmedNameListMap = Util.setMap(self.cuiToStemmedNameListMap, cui, stemmedConceptName)

        conceptNameTokens = re.split(r"\s+", conceptName)
        for conceptNameToken in conceptNameTokens:
            if conceptNameToken in Ling.getStopwordsList():
                continue

            self.tokenToNameListMap = Util.setMap(self.tokenToNameListMap, conceptNameToken, conceptName);



    def loadTerminology(self, dictionary, isMeddra_dict):

        # with codecs.open(path, 'r', 'UTF-8') as fp:
        #     for line in fp:
        #         line = line.strip()
        #         if line == u'':
        #             continue
        #         token = re.split(r"\|\|", line)
        #         cui = token[0]
        #
        #         conceptNames = token[1].lower()
        #
        #         self.loadMaps(conceptNames, cui)

        if isMeddra_dict:
            for cui, conceptNames in dictionary.items():
                self.loadMaps(conceptNames.lower(), cui)

        else:
            for cui, concept in dictionary.items():

                for concept_name in concept.names:

                    self.loadMaps(concept_name.lower(), cui)


    def clearTerminology(self):
        self.cuiAlternateCuiMap.clear()
        self.nameToCuiListMap.clear()
        self.cuiToNameListMap.clear()
        self.stemmedNameToCuiListMap.clear()
        self.cuiToStemmedNameListMap.clear()
        self.tokenToNameListMap.clear()
        self.compoundNameToCuiListMap.clear()
        self.simpleNameToCuiListMap.clear()

    def loadTrainingDataTerminology(self, documents, dictionary_reverse, isMeddra_dict):

        for document in documents:

            for mention in document.entities:

                conceptName = mention.name.lower().strip()
                for idx, norm_id in enumerate(mention.norm_ids):

                    if isMeddra_dict:
                        self.loadMaps(conceptName, norm_id)
                        cui = norm_id

                        simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(re.split(r"\s+", conceptName))
                        for simpleConceptName in simpleConceptNames:
                            self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap, simpleConceptName,
                                                                      cui)
                    else:
                        if norm_id in dictionary_reverse:
                            cui = dictionary_reverse[norm_id]
                            self.loadMaps(conceptName, cui[0])

                            simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(
                                re.split(r"\s+", conceptName))
                            for simpleConceptName in simpleConceptNames:
                                self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap,
                                                                          simpleConceptName,
                                                                          cui[0])

    def loadTrainingDataTerminology_frompath(self, path, dictionary_reverse, isMeddra_dict):

        for input_file_name in os.listdir(path):
            if input_file_name.find(".xml") == -1:
                continue
            input_file_path = os.path.join(path, input_file_name)

            if isMeddra_dict:
                annotation_file = get_fda_file(input_file_path)

                for mention in annotation_file.mentions:
                    conceptName = mention.name.lower().strip()
                    for idx, norm_id in enumerate(mention.norm_ids):

                        self.loadMaps(conceptName, norm_id)

                        cui = norm_id

                        simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(re.split(r"\s+", conceptName))
                        for simpleConceptName in simpleConceptNames:
                            self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap, simpleConceptName, cui)

            else:

                annotation_file = get_bioc_file(input_file_path)
                bioc_passage = annotation_file[0].passages[0]

                for entity in bioc_passage.annotations:
                    if opt.types and (entity.infons['type'] not in opt.type_filter):
                        continue
                    conceptName = entity.text.lower().strip()
                    if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A') :

                        if entity.infons['SNOMED code'] in dictionary_reverse:
                            cui = dictionary_reverse[entity.infons['SNOMED code']]
                            self.loadMaps(conceptName, cui[0])

                            simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(
                                re.split(r"\s+", conceptName))
                            for simpleConceptName in simpleConceptNames:
                                self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap,
                                                                          simpleConceptName,
                                                                          cui[0])


                    elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A') :

                        if entity.infons['MedDRA code'] in dictionary_reverse:
                            cui = dictionary_reverse[entity.infons['MedDRA code']]
                            self.loadMaps(conceptName, cui[0])

                            simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(
                                re.split(r"\s+", conceptName))
                            for simpleConceptName in simpleConceptNames:
                                self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap,
                                                                          simpleConceptName,
                                                                          cui[0])



    def loadTAC2017Terminology(self, path, dictionary):
        for input_file_name in os.listdir(path):
            if input_file_name.find(".xml") == -1:
                continue
            input_file_path = os.path.join(path, input_file_name)

            annotation_file = get_fda_file(input_file_path)

            for reaction in annotation_file.reactions:

                conceptName = reaction.name.lower().strip()

                for normalization in reaction.normalizations:

                    if normalization.meddra_pt_id is None:
                        continue

                    if normalization.meddra_pt_id not in dictionary:
                        # logging.info(normalization.meddra_pt_id)
                        continue

                    # conceptName = normalization.meddra_pt.lower().strip()

                    self.loadMaps(conceptName, normalization.meddra_pt_id)

                    cui = normalization.meddra_pt_id

                    simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(re.split(r"\s+", conceptName))
                    for simpleConceptName in simpleConceptNames:
                        self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap, simpleConceptName, cui)

    @classmethod
    def getOMIMCuis(self, cuis):
        OMIMcuis = list()
        for cui in cuis:
            if cui.find(u"OMIM") == -1:
                continue
            cui = re.split(u":", cui)[1]
            OMIMcuis = Util.setList(OMIMcuis, cui)
        return OMIMcuis

    def setOMIM(self, cuis, MeSHorSNOMEDcuis, conceptName):
        if MeSHorSNOMEDcuis == u"": # 如果Mesh ID为空，则用OMIM
            cuis = cuis.replace(u"OMIM:", u"")
            self.loadMaps(conceptName, cuis)

        else : # 否则用OMIM为候选ID
            cuis_arr = re.split(r"\|", cuis)
            for cui in cuis_arr:
                if cui.find(u"OMIM") == -1:
                    continue
                cui = re.split(u":", cui)[1]
                self.cuiAlternateCuiMap = Util.setMap(self.cuiAlternateCuiMap, MeSHorSNOMEDcuis, cui)

class Concept:
    def __init__(self, indexes, name, goldMeSHorSNOMEDCui, goldOMIMCuis):
        self.indexes = indexes
        self.name = name.lower().strip()
        self.goldMeSHorSNOMEDCui = goldMeSHorSNOMEDCui
        self.goldOMIMCuis = goldOMIMCuis
        self.nameExpansion = None
        self.stemmedName = None
        self.cui = None
        self.alternateCuis = None
        self.normalizingSieveLevel = 0
        self.namesKnowledgeBase = list()
        self.stemmedNamesKnowledgeBase = list()

    def setNameExpansion(self, text, abbreviationObject):
        self.nameExpansion = Abbreviation.getAbbreviationExpansion(abbreviationObject, text, self.name, self.indexes)

    def setStemmedName(self):
        self.stemmedName = Ling.getStemmedPhrase(self.name)

    def setCui(self, cui):
        self.cui = cui

    def getCui(self):
        return self.cui

    def setAlternateCuis(self, alternateCuis):

        self.alternateCuis = list()
        for alternateCui in alternateCuis:
            self.alternateCuis = Util.setList(self.alternateCuis, alternateCui)

    def setNormalizingSieveLevel(self, sieveLevel):
        self.normalizingSieveLevel = sieveLevel

    def getName(self):

        return self.name

    def getNormalizingSieve(self):
        return self.normalizingSieveLevel

    def getGoldMeSHorSNOMEDCui(self):
        return self.goldMeSHorSNOMEDCui

    def getGoldOMIMCuis(self):
        return self.goldOMIMCuis

    def getAlternateCuis(self):
        return self.alternateCuis

    def getNameExpansion(self):
        return self.nameExpansion

    def setNamesKnowledgeBase(self, name):
        if isinstance(name, list):
            self.namesKnowledgeBase = Util.addUnique(self.namesKnowledgeBase, name)
        else:
            self.namesKnowledgeBase = Util.setList(self.namesKnowledgeBase, name)

    def getNamesKnowledgeBase(self):
        return self.namesKnowledgeBase


    def getStemmedNamesKnowledgeBase(self):
        return self.stemmedNamesKnowledgeBase

    def setStemmedNamesKnowledgeBase(self, namesList):
        self.stemmedNamesKnowledgeBase = Util.addUnique(self.stemmedNamesKnowledgeBase, namesList)


class Evaluation:
    totalNames = 0
    tp = 0
    fp = 0
    accuracy = 0.0
    map_whichSieveFires = dict()

    @classmethod
    def initialize(self, data):

        for i in range(int(data.config['norm_rule_num'])+1):
            Evaluation.map_whichSieveFires[i] = 0


    @classmethod
    def incrementTotal(self):
        Evaluation.totalNames += 1

    @classmethod
    def incrementTP(self):
        Evaluation.tp += 1

    @classmethod
    def incrementFP(self):
        Evaluation.fp += 1


    @classmethod
    def evaluateClassification(self, concept, concepts):
        Evaluation.incrementTotal()
        if (concept.getGoldMeSHorSNOMEDCui() != u"" and concept.getGoldMeSHorSNOMEDCui() == concept.getCui()) \
            or (len(concept.getGoldOMIMCuis()) != 0 and concept.getCui() in concept.getGoldOMIMCuis()):
            Evaluation.incrementTP()
        elif concept.getGoldMeSHorSNOMEDCui().find(u"|") != -1 and concept.getCui().find(u"|") != -1:
            gold = set(re.split(r"\|", concept.getGoldMeSHorSNOMEDCui()))
            predicted = set(re.split(r"\|", concept.getCui()))

            bFindPredictNotInGold = False
            for p in predicted:
                if p not in gold:
                    bFindPredictNotInGold = True
                    break
            if bFindPredictNotInGold:
                Evaluation.incrementFP()
            else:
                Evaluation.incrementTP()

            minus_set = gold - predicted
            if len(minus_set) == 0:
                Evaluation.incrementTP()
            else :
                Evaluation.incrementFP()

        elif concept.getAlternateCuis() is not None and len(concept.getAlternateCuis()) != 0 :
            if concept.getGoldMeSHorSNOMEDCui() != u"" and concept.getGoldMeSHorSNOMEDCui() in concept.getAlternateCuis() :
                Evaluation.incrementTP()
                concept.setCui(concept.getGoldMeSHorSNOMEDCui())

            elif len(concept.getGoldOMIMCuis()) != 0 and Util.containsAny(concept.getAlternateCuis(), concept.getGoldOMIMCuis()) :
                Evaluation.incrementTP();
                if len(concept.getGoldOMIMCuis()) == 1:
                    concept.setCui(concept.getGoldOMIMCuis()[0])

            else :
                Evaluation.incrementFP()
        else :
            Evaluation.incrementFP()

        count = Evaluation.map_whichSieveFires.get(concept.normalizingSieveLevel)
        count += 1
        Evaluation.map_whichSieveFires[concept.normalizingSieveLevel] = count

    @classmethod
    def computeAccuracy(self):

        Evaluation.accuracy = Evaluation.tp * 1.0 / Evaluation.totalNames

    @classmethod
    def printResults(self):

        print("*********************")
        print("Total Names: {}".format(Evaluation.totalNames))
        print("True Normalizations: {}".format(Evaluation.tp))
        print("False Normalizations: {}".format(Evaluation.fp))
        print("Accuracy: {}".format(Evaluation.accuracy))
        print("*********************")

        for sieve_level in Evaluation.map_whichSieveFires:
            if sieve_level == 0:
                print("{} unmapped names, accounting for {:.2f}%".format(Evaluation.map_whichSieveFires[sieve_level],
                                                                        Evaluation.map_whichSieveFires[sieve_level]*100.0/Evaluation.totalNames))
            else:
                print("Sieve {} fires {} times, accounting for {:.2f}%".format(sieve_level, Evaluation.map_whichSieveFires[sieve_level],
                                                                        Evaluation.map_whichSieveFires[sieve_level]*100.0/Evaluation.totalNames))

        print("*********************")





class MultiPassSieveNormalizer:
    maxSieveLevel = 0

    def __init__(self):
        pass

    @classmethod
    def pass_(self, concept, currentSieveLevel):
        if concept.getCui() != u"":
            concept.setAlternateCuis(Sieve.getAlternateCuis(concept.getCui()))
            concept.setNormalizingSieveLevel(currentSieveLevel-1)

            return False


        if currentSieveLevel > MultiPassSieveNormalizer.maxSieveLevel:
            return False

        return True

    @classmethod
    def applyMultiPassSieve(self, concept):
        currentSieveLevel = 1

        # Sieve 1
        concept.setCui(Sieve.exactMatchSieve(concept.getName()))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 2
        concept.setCui(Sieve.exactMatchSieve(concept.getNameExpansion()))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 3
        concept.setCui(PrepositionalTransformSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 4
        concept.setCui(SymbolReplacementSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 5
        concept.setCui(HyphenationSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 6
        concept.setCui(AffixationSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 7
        concept.setCui(DiseaseModifierSynonymsSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 8
        concept.setCui(StemmingSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 9
        concept.setCui(CompoundPhraseSieve.applyNCBI(concept.getName()))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 10
        concept.setCui(SimpleNameSieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return

        # Sieve 11
        concept.setCui(PartialMatchNCBISieve.apply(concept))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return



class Sieve:
    standardTerminology = Terminology()
    trainingDataTerminology = Terminology()
    tac2017Terminology = Terminology()
    use_tac2017Terminology = False

    @classmethod
    def setStandardTerminology(self, dictionary, isMeddra_dict):
        Sieve.standardTerminology.loadTerminology(dictionary, isMeddra_dict)

    @classmethod
    def clearStandardTerminology(self):
        Sieve.standardTerminology.clearTerminology()

    @classmethod
    def setTrainingDataTerminology(self, train_path, dictionary_reverse, isMeddra_dict):
        Sieve.trainingDataTerminology.loadTrainingDataTerminology(train_path, dictionary_reverse, isMeddra_dict)

    @classmethod
    def clearTrainingDataTerminology(self):
        Sieve.trainingDataTerminology.clearTerminology()

    @classmethod
    def setTrainingDataTerminology_frompath(self, train_path, dictionary_reverse, isMeddra_dict):
        Sieve.trainingDataTerminology.loadTrainingDataTerminology_frompath(train_path, dictionary_reverse, isMeddra_dict)

    @classmethod
    def setTAC2017Terminology(self, train_path, dictionary):
        Sieve.tac2017Terminology.loadTAC2017Terminology(train_path, dictionary)
        Sieve.use_tac2017Terminology = True

    @classmethod
    def clearTAC2017Terminology(self):
        Sieve.tac2017Terminology.clearTerminology()
        Sieve.use_tac2017Terminology = False

    @classmethod
    def getAlternateCuis(self, cui):
        alternateCuis = list()
        if cui in Sieve.trainingDataTerminology.getCuiAlternateCuiMap():
            alternateCuis.extend(Sieve.trainingDataTerminology.getCuiAlternateCuiMap().get(cui))

        if cui in Sieve.standardTerminology.getCuiAlternateCuiMap():
            alternateCuis.extend(Sieve.standardTerminology.getCuiAlternateCuiMap().get(cui))

        if Sieve.use_tac2017Terminology:
            if cui in Sieve.tac2017Terminology.getCuiAlternateCuiMap():
                alternateCuis.extend(Sieve.tac2017Terminology.getCuiAlternateCuiMap().get(cui))

        return alternateCuis

    @classmethod
    def getTerminologyNameCui(self, nameToCuiListMap, name):
        return nameToCuiListMap.get(name)[0] if name in nameToCuiListMap and len(nameToCuiListMap.get(name)) == 1 else u""


    @classmethod
    def exactMatchSieve(self, name):
        cui = u""
        # check against names in the training data
        cui = Sieve.getTerminologyNameCui(Sieve.trainingDataTerminology.getNameToCuiListMap(), name)
        if cui != u"":
            return cui

        # check against names in the dictionary
        cui = Sieve.getTerminologyNameCui(Sieve.standardTerminology.getNameToCuiListMap(), name)
        if cui != u"":
            return cui

        if Sieve.use_tac2017Terminology:
            cui = Sieve.getTerminologyNameCui(Sieve.tac2017Terminology.getNameToCuiListMap(), name)
            if cui != u"":
                return cui

        return cui


    @classmethod
    def getTrainingDataTerminology(self):
        return Sieve.trainingDataTerminology

    @classmethod
    def getTAC2017Terminology(self):
        return Sieve.tac2017Terminology

    @classmethod
    def normalize(self, namesKnowledgeBase):
        for name in namesKnowledgeBase :
            cui = Sieve.exactMatchSieve(name)
            if cui != u"":
                return cui

        return u""

    @classmethod
    def getStandardTerminology(self):
        return Sieve.standardTerminology


class PrepositionalTransformSieve(Sieve):

    @classmethod
    def apply(self, concept):
        PrepositionalTransformSieve.init(concept)
        PrepositionalTransformSieve.transformName(concept)
        return Sieve.normalize(concept.getNamesKnowledgeBase())

    @classmethod
    def init(self, concept):
        concept.setNamesKnowledgeBase(concept.getName())
        if concept.getNameExpansion() != u"":
            concept.setNamesKnowledgeBase(concept.getNameExpansion())

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            prepositionInName = Ling.getStringPreposition(nameForTransformation)

            if prepositionInName != u"":
                transformedNames = Util.addUnique(transformedNames, PrepositionalTransformSieve.substitutePrepositionsInPhrase(prepositionInName, nameForTransformation))
                transformedNames = Util.setList(transformedNames, PrepositionalTransformSieve.swapPhrasalSubjectAndObject(prepositionInName, re.split(r"\s+", nameForTransformation)))
            else :
                transformedNames = Util.addUnique(transformedNames, PrepositionalTransformSieve.insertPrepositionsInPhrase(nameForTransformation, re.split(r"\s+", nameForTransformation)))

        concept.setNamesKnowledgeBase(transformedNames)

    @classmethod
    def insertPrepositionsInPhrase(self, phrase, phraseTokens):

        newPrepositionalPhrases = list()
        for preposition in Ling.PREPOSITIONS:

            newPrepositionalPhrase = (Ling.getSubstring(phraseTokens, 1, len(phraseTokens))+u" "+preposition+u" "+phraseTokens[0]).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)

            newPrepositionalPhrase = (phraseTokens[len(phraseTokens)-1]+u" "+preposition+u" "+Ling.getSubstring(phraseTokens, 0, len(phraseTokens)-1)).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)

        return newPrepositionalPhrases



    @classmethod
    def substitutePrepositionsInPhrase(self, prepositionInPhrase, phrase):
        newPrepositionalPhrases = list()
        for preposition in Ling.PREPOSITIONS:
            if preposition == prepositionInPhrase:
                continue

            newPrepositionalPhrase = (phrase.replace(u" " + prepositionInPhrase + u" ", u" " + preposition + u" ")).strip()
            newPrepositionalPhrases = Util.setList(newPrepositionalPhrases, newPrepositionalPhrase)

        return newPrepositionalPhrases

    @classmethod
    def swapPhrasalSubjectAndObject(self, prepositionInPhrase, phraseTokens) :
        prepositionTokenIndex = Util.getTokenIndex(phraseTokens, prepositionInPhrase)
        return  (Ling.getSubstring(phraseTokens, prepositionTokenIndex+1, len(phraseTokens))+u" "+
                Ling.getSubstring(phraseTokens, 0, prepositionTokenIndex)).strip() if prepositionTokenIndex != -1 else u""


class SymbolReplacementSieve(Sieve):

    @classmethod
    def apply(self, concept):
        SymbolReplacementSieve.transformName(concept)
        return Sieve.normalize(concept.getNamesKnowledgeBase())

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, SymbolReplacementSieve.substituteSymbolsInStringWithWords(nameForTransformation))
            transformedNames = Util.addUnique(transformedNames, SymbolReplacementSieve.substituteWordsInStringWithSymbols(nameForTransformation))


        concept.setNamesKnowledgeBase(transformedNames)

    @classmethod
    def getClinicalReportTypeSubstitutions(self, string):
        newStrings = list()
        for digit in Ling.getDigitToWordMap():
            if string.find(digit) == -1:
                continue
            wordsList = Ling.getDigitToWordMap().get(digit)
            for word in wordsList:
                newString = string.replace(digit, word)
                if newString != string:
                    newStrings = Util.setList(newStrings, newString)

        return newStrings

    @classmethod
    def getBiomedicalTypeSubstitutions(self, string):
        if string.find(u"and/or") != -1:
            string = string.replace(u"and/or", u"and")
        if string.find(u"/") != -1:
            string = string.replace(u"/", u" and ")
        if string.find(u" (") != -1 and string.find(u")") != -1:
            string = string.replace(u" (", u"").replace(u")", u"")
        elif string.find(u"(") != -1 and string.find(u")") != -1:
            string = string.replace(u"(", u"").replace(u")", u"")
        return string



    @classmethod
    def substituteSymbolsInStringWithWords(self, string):
        newStrings = SymbolReplacementSieve.getClinicalReportTypeSubstitutions(string)
        tempNewStrings = list()
        for newString in newStrings:
            tempNewStrings = Util.setList(tempNewStrings, SymbolReplacementSieve.getBiomedicalTypeSubstitutions(newString))
        newStrings = Util.addUnique(newStrings, tempNewStrings)
        newStrings = Util.setList(newStrings, SymbolReplacementSieve.getBiomedicalTypeSubstitutions(string))
        return newStrings

    @classmethod
    def substituteWordsInStringWithSymbols(self, string):
        newStrings = list()
        for word in Ling.getWordToDigitMap():
            if string.find(word) == -1:
                continue
            digit = Ling.getWordToDigitMap().get(word)
            newString = string.replace(word, digit)
            if newString != string:
                newStrings = Util.setList(newStrings, newString)

        return newStrings


class HyphenationSieve(Sieve):
    @classmethod
    def apply(self, concept):
        HyphenationSieve.transformName(concept)
        return Sieve.normalize(concept.getNamesKnowledgeBase())

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, HyphenationSieve.hyphenateString(re.split(r"\s+", nameForTransformation)))
            transformedNames = Util.addUnique(transformedNames, HyphenationSieve.dehyphenateString(re.split(r"\-", nameForTransformation)))


        concept.setNamesKnowledgeBase(transformedNames)

    @classmethod
    def hyphenateString(self, stringTokens):
        hyphenatedStrings = list()
        i = 1
        while i < len(stringTokens):
            hyphenatedString = u""
            j = 0
            while j < len(stringTokens):
                if j == i:
                    hyphenatedString += u"-"+stringTokens[j]
                else:
                    hyphenatedString = stringTokens[j] if hyphenatedString == u"" else hyphenatedString+u" "+stringTokens[j]
                j += 1

            hyphenatedStrings = Util.setList(hyphenatedStrings, hyphenatedString)
            i += 1
        return hyphenatedStrings

    @classmethod
    def dehyphenateString(self, stringTokens):
        dehyphenatedStrings = list()
        i = 1
        while i < len(stringTokens):

            dehyphenatedString = u""
            j = 0
            while j < len(stringTokens):
                if j == i:
                    dehyphenatedString += u" "+stringTokens[j]
                else:
                    dehyphenatedString = stringTokens[j] if dehyphenatedString == u"" else dehyphenatedString+u"-"+stringTokens[j]
                j += 1

            dehyphenatedStrings = Util.setList(dehyphenatedStrings, dehyphenatedString)
            i += 1

        return dehyphenatedStrings

class AffixationSieve(Sieve):

    @classmethod
    def apply(self, concept):
        AffixationSieve.transformName(concept)
        return Sieve.normalize(concept.getNamesKnowledgeBase())

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, AffixationSieve.affix(nameForTransformation))


        concept.setNamesKnowledgeBase(transformedNames);

    @classmethod
    def getAllStringTokenSuffixationCombinations(self, stringTokens):
        suffixatedPhrases = list()
        for stringToken in stringTokens:
            suffix = Ling.getSuffix(stringToken)
            forSuffixation = None if suffix == u"" else Ling.getSuffixMap().get(suffix)

            if len(suffixatedPhrases) == 0:
                if forSuffixation is None:
                    suffixatedPhrases = Util.setList(suffixatedPhrases, stringToken)
                elif len(forSuffixation) == 0:
                    suffixatedPhrases = Util.setList(suffixatedPhrases, stringToken.replace(suffix, u""))
                else :
                    i = 0
                    while i < len(forSuffixation):
                        suffixatedPhrases = Util.setList(suffixatedPhrases, stringToken.replace(suffix, forSuffixation[i]))
                        i += 1

            else :
                if forSuffixation is None:
                    for i in range(len(suffixatedPhrases)):
                        suffixatedPhrases[i] = suffixatedPhrases[i]+u" "+stringToken

                elif len(forSuffixation) == 0:
                    for i in range(len(suffixatedPhrases)):
                        suffixatedPhrases[i] = suffixatedPhrases[i]+u" "+stringToken.replace(suffix, u"")

                else:
                    tempSuffixatedPhrases = list()
                    for i in range(len(suffixatedPhrases)):
                        suffixatedPhrase = suffixatedPhrases[i]
                        for j in range(len(forSuffixation)):
                            tempSuffixatedPhrases = Util.setList(tempSuffixatedPhrases, suffixatedPhrase+u" "+stringToken.replace(suffix, forSuffixation[j]))

                    suffixatedPhrases = list(tempSuffixatedPhrases)
                    tempSuffixatedPhrases = None


        return suffixatedPhrases

    @classmethod
    def getUniformStringTokenSuffixations(self, stringTokens, string):
        suffixatedPhrases = list()
        for stringToken in stringTokens:
            suffix = Ling.getSuffix(stringToken)
            forSuffixation = None if suffix == u"" else Ling.getSuffixMap().get(suffix)

            if forSuffixation == None:
                continue

            if len(forSuffixation) == 0:
                Util.setList(suffixatedPhrases, string.replace(suffix, u""))
                continue

            for i in range(len(forSuffixation)):
                suffixatedPhrases = Util.setList(suffixatedPhrases, string.replace(suffix, forSuffixation[i]))

        return suffixatedPhrases


    @classmethod
    def suffixation(self, stringTokens, string):
        suffixatedPhrases = AffixationSieve.getAllStringTokenSuffixationCombinations(stringTokens)
        return Util.addUnique(suffixatedPhrases, AffixationSieve.getUniformStringTokenSuffixations(stringTokens, string))

    @classmethod
    def prefixation(self, stringTokens, string):
        prefixatedPhrase = u""
        for stringToken in stringTokens:
            prefix = Ling.getPrefix(stringToken)
            forPrefixation = u"" if prefix == u"" else Ling.getPrefixMap().get(prefix)
            if prefixatedPhrase == u"":
                prefixatedPhrase = stringToken if prefix == u"" else stringToken.replace(prefix, forPrefixation)
            else:
                prefixatedPhrase = prefixatedPhrase+u" "+stringToken if prefix == u"" else prefixatedPhrase+u" "+stringToken.replace(prefix, forPrefixation)

        return prefixatedPhrase

    @classmethod
    def affixation(self, stringTokens, string):
        affixatedPhrase = u""
        for stringToken in stringTokens:
            affix = (re.split(r"\|", Ling.AFFIX)[0] if stringToken.find(re.split(r"\|",Ling.AFFIX)[0]) != -1 else re.split(r"\|", Ling.AFFIX)[1]) \
                if re.match(r".*("+Ling.AFFIX+r").*", stringToken) else u""
            forAffixation = u"" if affix == u"" else Ling.getAffixMap().get(affix)
            if affixatedPhrase == u"":
                affixatedPhrase = stringToken if affix == u"" else stringToken.replace(affix, forAffixation)
            else:
                affixatedPhrase = affixatedPhrase+u" "+stringToken if affix == u"" else affixatedPhrase+u" "+stringToken.replace(affix, forAffixation)

        return affixatedPhrase

    @classmethod
    def affix(self, string):
        stringTokens = re.split(r"\s", string)
        newPhrases = AffixationSieve.suffixation(stringTokens, string)
        newPhrases = Util.setList(newPhrases, AffixationSieve.prefixation(stringTokens, string))
        newPhrases = Util.setList(newPhrases, AffixationSieve.affixation(stringTokens, string))
        return newPhrases


class DiseaseModifierSynonymsSieve(Sieve):

    @classmethod
    def apply(self, concept):
        if concept.getName() not in Ling.PLURAL_DISORDER_SYNONYMS and concept.getName() not in Ling.SINGULAR_DISORDER_SYNONYMS:
            DiseaseModifierSynonymsSieve.transformName(concept)
            return Sieve.normalize(concept.getNamesKnowledgeBase())

        return u""

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            nameForTransformationTokens = re.split(r"\s+", nameForTransformation)
            modifier = DiseaseModifierSynonymsSieve.getModifier(nameForTransformationTokens, Ling.PLURAL_DISORDER_SYNONYMS)
            if modifier != u"":
                transformedNames = Util.addUnique(transformedNames, DiseaseModifierSynonymsSieve.substituteDiseaseModifierWithSynonyms(nameForTransformation, modifier, Ling.PLURAL_DISORDER_SYNONYMS))
                transformedNames = Util.setList(transformedNames, DiseaseModifierSynonymsSieve.deleteTailModifier(nameForTransformationTokens, modifier))
                continue


            modifier = DiseaseModifierSynonymsSieve.getModifier(nameForTransformationTokens, Ling.SINGULAR_DISORDER_SYNONYMS)
            if modifier != u"":
                transformedNames = Util.addUnique(transformedNames, DiseaseModifierSynonymsSieve.substituteDiseaseModifierWithSynonyms(nameForTransformation, modifier, Ling.SINGULAR_DISORDER_SYNONYMS))
                transformedNames = Util.setList(transformedNames, DiseaseModifierSynonymsSieve.deleteTailModifier(nameForTransformationTokens, modifier))
                continue

            transformedNames = Util.addUnique(transformedNames, DiseaseModifierSynonymsSieve.appendModifier(nameForTransformation, Ling.SINGULAR_DISORDER_SYNONYMS))


        concept.setNamesKnowledgeBase(transformedNames);

    @classmethod
    def substituteDiseaseModifierWithSynonyms(self, string, toReplaceWord, synonyms):
        newPhrases = list()
        for synonym in synonyms:
            if toReplaceWord == synonym:
                continue
            newPhrase = string.replace(toReplaceWord, synonym)
            newPhrases = Util.setList(newPhrases, newPhrase)

        return newPhrases

    @classmethod
    def deleteTailModifier(self, stringTokens, modifier):
        return Ling.getSubstring(stringTokens, 0, len(stringTokens) - 1) if stringTokens[len(stringTokens) - 1] == modifier else u""

    @classmethod
    def appendModifier(self, string, modifiers):
        newPhrases = list()
        for modifier in modifiers:
            newPhrase = string + u" " + modifier
            newPhrases = Util.setList(newPhrases, newPhrase)

        return newPhrases


    @classmethod
    def getModifier(self, stringTokens, modifiers):
        for modifier in modifiers:
            index = Util.getTokenIndex(stringTokens, modifier)
            if index != -1:
                return stringTokens[index]

        return u""


class StemmingSieve(Sieve):
    @classmethod
    def apply(self, concept):
        StemmingSieve.transformName(concept)
        return StemmingSieve.normalize(concept)

    @classmethod
    def transformName(self, concept):
        namesForTransformation = list(concept.getNamesKnowledgeBase())
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            transformedNames = Util.setList(transformedNames, Ling.getStemmedPhrase(nameForTransformation))


        concept.setStemmedNamesKnowledgeBase(transformedNames)

    @classmethod
    def normalize(self, concept):
        for name in concept.getStemmedNamesKnowledgeBase():
            cui = StemmingSieve.exactMatchSieve(name)
            if cui != u"":
                return cui

        return u""

    @classmethod
    def exactMatchSieve(self, name):
        cui = u""

        # checks against names in training data
        cui = Sieve.getTerminologyNameCui(Sieve.getTrainingDataTerminology().getStemmedNameToCuiListMap(), name)
        if cui != u"":
            return cui

        # checks against names in dictionary
        cui = Sieve.getTerminologyNameCui(Sieve.getStandardTerminology().getStemmedNameToCuiListMap(), name)
        if cui != u"":
            return cui

        if Sieve.use_tac2017Terminology:
            cui = Sieve.getTerminologyNameCui(Sieve.getTAC2017Terminology().getStemmedNameToCuiListMap(), name)
            if cui != u"":
                return cui

        return cui




class CompoundPhraseSieve(Sieve):

    @classmethod
    def applyNCBI(self, name):
        cui = CompoundPhraseSieve.apply(name)
        if cui != u"" or (name.find(u" and ") == -1 and name.find(u" or ") == -1):
            return cui

        compoundWord = u"and" if name.find(u" and ") else u"or"
        nameTokens = re.split(r"\s+", name)
        index = Util.getTokenIndex(nameTokens, compoundWord)

        if index == 1:
            replacement1 = nameTokens[0]
            replacement2 = nameTokens[2]+u" "+nameTokens[3] if nameTokens[2] == u"the" else nameTokens[2]
            phrase = replacement1+u" "+compoundWord+u" "+replacement2
            replacement2 = nameTokens[3] if nameTokens[2] == u"the" else nameTokens[2]
            cui1 = Sieve.exactMatchSieve(name.replace(phrase, replacement1))

            cui2 = Sieve.exactMatchSieve(name.replace(phrase, replacement2))
            if cui1 != u"" and cui2 != u"":
                return cui2+u"|"+cui1 if cui2+u"|"+cui1 in Sieve.getTrainingDataTerminology().getCuiToNameListMap() else cui1+u"|"+cui2


        return u""

    @classmethod
    def apply(self, name):
        cui = u""

        cui = Sieve.getTerminologyNameCui(Sieve.getTrainingDataTerminology().getCompoundNameToCuiListMap(), name)
        if cui != u"":
            return cui

        cui = Sieve.getTerminologyNameCui(Sieve.getStandardTerminology().getCompoundNameToCuiListMap(), name)
        if cui != u"":
            return cui

        if Sieve.use_tac2017Terminology:
            cui = Sieve.getTerminologyNameCui(Sieve.getTAC2017Terminology().getCompoundNameToCuiListMap(), name)
            if cui != u"":
                return cui

        return cui


class SimpleNameSieve(Sieve):

    @classmethod
    def apply(self, concept):
        namesForTransformation = SimpleNameSieve.getNamesForTransformation(concept)
        namesKnowledgeBase = SimpleNameSieve.transformName(namesForTransformation)
        cui = Sieve.normalize(namesKnowledgeBase)
        return SimpleNameSieve.normalize(concept.getName()) if cui == u"" else cui

    @classmethod
    def getNamesForTransformation(self, concept):
        namesForTransformation = list()
        namesForTransformation.append(concept.getName())
        if concept.getNameExpansion() != u"":
            namesForTransformation.append(concept.getNameExpansion())
        return namesForTransformation

    @classmethod
    def transformName(self, namesForTransformation):
        transformedNames = list()

        for nameForTransformation in namesForTransformation:
            transformedNames = Util.addUnique(transformedNames, SimpleNameSieve.deletePhrasalModifier(nameForTransformation, re.split(r"\s", nameForTransformation)))


        return transformedNames

    @classmethod
    def deletePhrasalModifier(self, phrase, phraseTokens):
        newPhrases = list()
        if len(phraseTokens) > 3:
            newPhrase = Ling.getSubstring(phraseTokens, 0, len(phraseTokens)-2)+u" "+phraseTokens[len(phraseTokens)-1]
            newPhrases = Util.setList(newPhrases, newPhrase)
            newPhrase = Ling.getSubstring(phraseTokens, 1, len(phraseTokens))
            newPhrases = Util.setList(newPhrases, newPhrase)

        return newPhrases


    @classmethod
    def getTerminologySimpleNames(self, phraseTokens):
        newPhrases = list()
        if len(phraseTokens) == 3 :
            newPhrase = phraseTokens[0]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)
            newPhrase = phraseTokens[1]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)

        return newPhrases

    @classmethod
    def normalize(self, name):
        cui = u""

        cui = Sieve.getTerminologyNameCui(Sieve.getTrainingDataTerminology().getSimpleNameToCuiListMap(), name)
        if cui != u"":
            return cui

        if Sieve.use_tac2017Terminology:
            cui = Sieve.getTerminologyNameCui(Sieve.getTAC2017Terminology().getSimpleNameToCuiListMap(), name)
            if cui != u"":
                return cui

        return cui


class PartialMatchNCBISieve:
    @classmethod
    def apply(self, concept):
        name = concept.getName()
        nameTokens = re.split(r"\s+", name)
        return PartialMatchNCBISieve.partialMatch(name, nameTokens)

    @classmethod
    def partialMatch(self, phrase, phraseTokens):
        partialMatchedPhrases = list()
        candidateCuiDataMap = PartialMatchNCBISieve.init()

        for phraseToken in phraseTokens:
            if phraseToken in Ling.getStopwordsList():
                continue
            candidatePhrases = None
            map = -1

            if phraseToken in Sieve.getTrainingDataTerminology().getTokenToNameListMap():
                candidatePhrases = list(Sieve.getTrainingDataTerminology().getTokenToNameListMap().get(phraseToken))
                map = 2

            elif phraseToken in Sieve.getStandardTerminology().getTokenToNameListMap():
                candidatePhrases = list(Sieve.getStandardTerminology().getTokenToNameListMap().get(phraseToken))
                map = 3


            if candidatePhrases is None:
                continue

            temp = list()
            for t in candidatePhrases:
                if t in partialMatchedPhrases:
                    continue
                temp.append(t)
            candidatePhrases = temp

            candidateCuiDataMap = PartialMatchNCBISieve.ncbiPartialMatch(phrase, candidatePhrases, partialMatchedPhrases, Sieve.getTrainingDataTerminology() if map == 2 else Sieve.getStandardTerminology(), candidateCuiDataMap)

        return PartialMatchNCBISieve.getCui(candidateCuiDataMap.get(1), candidateCuiDataMap.get(2)) if len(candidateCuiDataMap.get(1)) != 0 else u""

    @classmethod
    def init(self):
        candidateCuiDataMap = dict()
        candidateCuiDataMap[1] = dict()
        candidateCuiDataMap[2] = dict()
        return candidateCuiDataMap

    @classmethod
    def ncbiPartialMatch(self, phrase, candidatePhrases, partialMatchedPhrases, terminology, cuiCandidateDataMap):
        cuiCandidateMatchingTokensCountMap = cuiCandidateDataMap.get(1)
        cuiCandidateLengthMap = cuiCandidateDataMap.get(2)

        for candidatePhrase in candidatePhrases:
            partialMatchedPhrases = Util.setList(partialMatchedPhrases, candidatePhrase)

            count = Ling.getMatchingTokensCount(phrase, candidatePhrase)
            length = len(re.split(r"\s+", candidatePhrase))
            cui = terminology.getNameToCuiListMap().get(candidatePhrase)[0]

            if cui in cuiCandidateMatchingTokensCountMap:
                oldCount = cuiCandidateMatchingTokensCountMap.get(cui)
                if oldCount < count:
                    cuiCandidateMatchingTokensCountMap[cui] =  count
                    cuiCandidateLengthMap[cui] = length

                continue


            cuiCandidateMatchingTokensCountMap[cui] = count
            cuiCandidateLengthMap[cui] = length


        cuiCandidateDataMap[1] = cuiCandidateMatchingTokensCountMap
        cuiCandidateDataMap[2] = cuiCandidateLengthMap
        return cuiCandidateDataMap

    @classmethod
    def getCui(self, cuiCandidateMatchedTokensCountMap, cuiCandidateLengthMap):
        cui = u""
        maxMatchedTokensCount = -1
        matchedTokensCountCuiListMap = dict()
        for candidateCui in cuiCandidateMatchedTokensCountMap:
            matchedTokensCount = cuiCandidateMatchedTokensCountMap.get(candidateCui)
            if matchedTokensCount >= maxMatchedTokensCount:
                maxMatchedTokensCount = matchedTokensCount

                cuiList = matchedTokensCountCuiListMap.get(matchedTokensCount)
                if cuiList is None:
                    cuiList = list()
                    matchedTokensCountCuiListMap[matchedTokensCount] = cuiList
                cuiList = Util.setList(cuiList, candidateCui)


        candidateCuiList = matchedTokensCountCuiListMap.get(maxMatchedTokensCount)
        if len(candidateCuiList) == 1:
            return candidateCuiList[0]
        else :
            minCandidateLength = 1000
            for candidateCui in candidateCuiList:
                length = cuiCandidateLengthMap.get(candidateCui)
                if length < minCandidateLength:
                    minCandidateLength = length
                    cui = candidateCui



        return cui


def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

from alphabet import Alphabet

multi_sieve_dict_alphabet = Alphabet('dict')

def init(opt, train_data, d, dictionary, dictionary_reverse, isMeddra_dict):
    logging.info("initialize the rule-based normalization model ...")

    Ling.setStopwordsList(os.path.join(d.config['norm_rule_resource'], 'stopwords.txt'))
    Abbreviation.setWikiAbbreviationExpansionMap(os.path.join(d.config['norm_rule_resource'], 'ncbi-wiki-abbreviations.txt'))
    Ling.setDigitToWordformMapAndReverse(os.path.join(d.config['norm_rule_resource'], 'number.txt'))
    Ling.setSuffixMap(os.path.join(d.config['norm_rule_resource'], 'suffix.txt'))
    Ling.setPrefixMap(os.path.join(d.config['norm_rule_resource'], 'prefix.txt'))
    Ling.setAffixMap(os.path.join(d.config['norm_rule_resource'], 'affix.txt'))


    MultiPassSieveNormalizer.maxSieveLevel = int(d.config['norm_rule_num'])

    Evaluation.initialize(d)

    Sieve.setStandardTerminology(dictionary, isMeddra_dict)

    if d.config.get('norm_rule_use_trainset') != '0':
        if train_data is None:
            if isMeddra_dict:
                Sieve.setTrainingDataTerminology_frompath(opt.train_file, dictionary_reverse, isMeddra_dict)
            else:
                Sieve.setTrainingDataTerminology_frompath(os.path.join(opt.train_file, "bioc"), dictionary_reverse, isMeddra_dict)
        else:
            Sieve.setTrainingDataTerminology(train_data, dictionary_reverse, isMeddra_dict)

    # external corpus
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                Sieve.setTAC2017Terminology(v['path'], dictionary)
            else:
                raise RuntimeError("wrong configuration")


    if multi_sieve_dict_alphabet.keep_growing:
        norm_utils.init_dict_alphabet(multi_sieve_dict_alphabet, dictionary)
        norm_utils.fix_alphabet(multi_sieve_dict_alphabet)


def runMultiPassSieve(document, entities, dictionary, isMeddra_dict):

    concepts = list()

    abbreviationObject = Abbreviation()
    abbreviationObject.setTextAbbreviationExpansionMap(document.text)

    for entity in entities:

        try:
            concept = Concept(str(entity.spans[0][0]) + "|" + str(entity.spans[0][1]), entity.name, None, None)
            concept.setNameExpansion(document.text, abbreviationObject)
            concept.setStemmedName()
            concepts.append(concept)

            MultiPassSieveNormalizer.applyMultiPassSieve(concept)
            if concept.getCui() == u"":
                concept.setCui(u"CUI-less")
        except Exception as e:
            logging.info("error when process {} in {}".format(entity.name, document.name))
            concept = Concept(str(entity.spans[0][0]) + "|" + str(entity.spans[0][1]), entity.name, None, None)
            concept.setCui(u"CUI-less")
            concepts.append(concept)
            continue


    # fill norm name and id into entity
    for idx, entity in enumerate(entities):
        id = concepts[idx].getCui()
        if id != u"CUI-less":
            for _id in id.split("|"):
                if isMeddra_dict:
                    name = dictionary[_id]
                    entity.norm_ids.append(_id)
                    entity.norm_names.append(name)
                else:
                    concept = dictionary[_id]
                    entity.norm_ids.append(_id)
                    entity.norm_names.append(concept.names)

                if opt.ensemble == 'sum':
                    confidences = np.zeros([len(dictionary)])
                    confidences[norm_utils.get_dict_index(multi_sieve_dict_alphabet, _id)] = 1
                    entity.norm_confidences.append(confidences)
                if entity.rule_id is None:
                    entity.rule_id = _id



def finalize(shutdownjvm):
    Ling.clearStopwordsList()
    Abbreviation.clearWikiAbbreviationExpansionMap()
    Ling.clearDigitToWordformMapAndReverse()
    Ling.clearSuffixMap()
    Ling.clearPrefixMap()
    Ling.clearAffixMap()

    Sieve.clearStandardTerminology()

    Sieve.clearTrainingDataTerminology()

    Sieve.clearTAC2017Terminology()

    if shutdownjvm:
        shutdownJVM()

def train(train_data, dev_data, d, dictionary, dictionary_reverse, opt, fold_idx, isMeddra_dict):

    init(opt, train_data, d, dictionary, dictionary_reverse, isMeddra_dict)

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    if opt.dev_file:
        p, r, f = norm_utils.evaluate(dev_data, dictionary, dictionary_reverse, None, None, None, d, isMeddra_dict)
        logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
    else:
        f = best_dev_f

    if f > best_dev_f:
        logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

        best_dev_f = f
        best_dev_p = p
        best_dev_r = r


    logging.info("train finished")

    if fold_idx is None:
        finalize(True)
    else:
        if fold_idx == opt.cross_validation-1:
            finalize(True)
        else:
            finalize(False)

    return best_dev_p, best_dev_r, best_dev_f

