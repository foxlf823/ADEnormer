
class Entity:
    def __init__(self):
        self.id = None
        self.type = None

        self.spans = [] # a couple of spans, list (start, end)
        self.tkSpans = []
        self.labelSpans = []
        self.name = None

        self.sent_idx = None
        self.norm_ids = []
        self.norm_names = []
        self.norm_confidences = []

        # for FDA challenge
        self.section = None

        # for ensemble
        self.rule_id = None
        self.vsm_id = None
        self.neural_id = None

    # def create(self, id, type, start, end, text, sent_idx, tf_start, tf_end):
    #     self.id = id
    #     self.type = type
    #     self.start = start
    #     self.end = end
    #     self.text = text
    #     self.sent_idx = sent_idx
    #     self.tf_start = tf_start
    #     self.tf_end = tf_end
    #
    # def append(self, start, end, text, tf_end):
    #
    #     whitespacetoAdd = start - self.end
    #     for _ in range(whitespacetoAdd):
    #         self.text += " "
    #     self.text += text
    #
    #     self.end = end
    #     self.tf_end = tf_end
    #
    # def getlength(self):
    #     return self.end-self.start

    def equals(self, other):

        if self.type == other.type and len(self.spans) == len(other.spans) :

            for i in range(len(self.spans)) :

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True
        else:
            return False

    def equals_span(self, other):
        if len(self.spans) == len(other.spans):

            for i in range(len(self.spans)):

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True

        else:
            return False

    def equalsTkSpan(self, other):
        if len(self.tkSpans) == len(other.tkSpans):

            for i in range(len(self.tkSpans)):

                if self.tkSpans[i][0] != other.tkSpans[i][0] or self.tkSpans[i][1] != other.tkSpans[i][1]:
                    return False

            return True

        else:
            return False



class Document:
    def __init__(self):
        self.entities = None
        self.sentences = None
        self.name = None
        self.text = None


# used for FDA challenge
class Section:
    def __init__(self):
        self.id = None
        self.name = None
        self.text = ""

class IgnoredRegion:
    def __init__(self):
        self.name = None
        self.section = None
        self.start = None
        self.end = None

# for tac 2017
class Reaction:
    def __init__(self):
        self.id = None
        self.name = None
        self.normalizations = []

class Normalization:
    def __init__(self):
        self.id = None
        self.meddra_pt = None
        self.meddra_pt_id = None
        self.meddra_llt = None
        self.meddra_llt_id = None



