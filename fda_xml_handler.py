
import xml.sax
import data_structure

class FdaXmlHandler( xml.sax.ContentHandler ):

    def __init__(self):
        self.currentTag = ""
        self.parentTag = []

        self.sections = []
        self.ignore_regions = []
        self.mentions = []

        # for tac 2017
        self.reactions = []

    def startDocument(self):
        pass

    def endDocument(self):
        self.currentTag = ""
        self.parentTag = []

    def startElement(self, tag, attributes):
        if self.currentTag != '':
            self.parentTag.append(self.currentTag)

        self.currentTag = tag

        if tag == 'Section':
            section = data_structure.Section()
            section.id = attributes['id']
            section.name = attributes['name']
            self.sections.append(section)
        elif tag == 'IgnoredRegion':
            ignored_region = data_structure.IgnoredRegion()
            ignored_region.name = attributes['name']
            ignored_region.section = attributes['section']
            ignored_region.start = int(attributes['start'])
            ignored_region.end = int(attributes['start'])+int(attributes['len'])
            self.ignore_regions.append(ignored_region)
        elif tag == 'Mention':
            mention = data_structure.Entity()
            mention.id = attributes['id']
            mention.section = attributes['section']
            mention.type = attributes['type']
            splitted_start = attributes['start'].split(',')
            splitted_len = attributes['len'].split(',')
            for i, _ in enumerate(splitted_start):
                mention.spans.append([int(splitted_start[i]), int(splitted_start[i])+int(splitted_len[i])])
            mention.name = ''
            mention_section = None
            for section in self.sections:
                if mention.section == section.id:
                    mention_section = section
                    break
            for span in mention.spans:
                mention.name += mention_section.text[span[0]:span[1]]+" "
            mention.name = mention.name.strip()
            self.mentions.append(mention)
        elif len(self.parentTag) > 0 and self.parentTag[-1] == 'Mention' and tag == 'Normalization': # for fda 2018
            current_mention = self.mentions[-1]
            meddra_pt_id = attributes.get('meddra_pt_id').strip() if attributes.get('meddra_pt_id') is not None else ''
            if meddra_pt_id != '':
                current_mention.norm_ids.append(attributes['meddra_pt_id'])
                current_mention.norm_names.append(attributes['meddra_pt'])
        elif tag == 'Reaction':
            reaction = data_structure.Reaction()
            reaction.id = attributes['id']
            reaction.name = attributes['str']
            self.reactions.append(reaction)
        elif len(self.parentTag) > 0 and self.parentTag[-1] == 'Reaction' and tag == 'Normalization': # for tac 2017
            current_reaction = self.reactions[-1]
            normalization = data_structure.Normalization()
            normalization.id = attributes['id']
            meddra_pt_id = attributes.get('meddra_pt_id').strip() if attributes.get('meddra_pt_id') is not None else ''
            if meddra_pt_id != '':
                normalization.meddra_pt =  attributes['meddra_pt']
                normalization.meddra_pt_id = attributes['meddra_pt_id']
            meddra_llt_id = attributes.get('meddra_llt_id').strip() if attributes.get('meddra_llt_id') is not None else ''
            if meddra_llt_id != '':
                normalization.meddra_llt = attributes['meddra_llt']
                normalization.meddra_llt_id = attributes['meddra_llt_id']
            current_reaction.normalizations.append(normalization)



    def endElement(self, tag):
        if len(self.parentTag) != 0:
            self.currentTag = self.parentTag[-1]
            self.parentTag.pop()
        else:
            self.currentTag = ''


    def characters(self, content):

        if self.currentTag == 'Section':
            if self.sections[-1].text is None:
                self.sections[-1].text = content
            else:
                self.sections[-1].text += content


if __name__ == '__main__':
    '/Users/feili/dataset/ADE Eval Shared Resources/ose_xml_training_20181101'