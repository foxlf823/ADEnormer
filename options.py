import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=1, help='1-train ner, 2-train norm, 3-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-train_file', default='./sample')
parser.add_argument('-dev_file', default='./sample')
parser.add_argument('-test_file', default='./sample')
parser.add_argument('-output', default='./output')
parser.add_argument('-iter', type=int, default=100)
parser.add_argument('-gpu', type=int, default=-1)
parser.add_argument('-tune_wordemb', action='store_true', default=False)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-word_emb_file', default=None)
parser.add_argument('-word_emb_dim', type=int, default=100)
parser.add_argument('-hidden_dim', type=int, default=100)
parser.add_argument('-char_emb_dim', type=int, default=24)
parser.add_argument('-char_hidden_dim', type=int, default=24)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0, help='0~0.5')
parser.add_argument('-l2', type=float, default=1e-8)
parser.add_argument('-nbest', type=int, default=0)
parser.add_argument('-patience', type=int, default=20)
parser.add_argument('-gradient_clip', type=float, default=5.0)
parser.add_argument('-types', default=None) # a,b,c
parser.add_argument('-predict', default='./predict')
parser.add_argument('-ner_number_normalized', action='store_true', default=False)
parser.add_argument('-norm_number_normalized', action='store_true', default=False)
parser.add_argument('-nlp_tool', default='nltk', help='spacy, nltk, stanford')
parser.add_argument('-no_type', action='store_true', default=False)
parser.add_argument('-test_in_cpu', action='store_true', default=False)
parser.add_argument('-schema', default='BMES', help='BMES, BIOHD_1234')
parser.add_argument('-cross_validation', type=int, default=1, help='1-not use cross validation; >1 - use n-fold cross validation')
parser.add_argument('-config', default='./config.txt')
parser.add_argument('-elmo', action='store_true', default=False)
parser.add_argument('-norm_rule', action='store_true', default=False)
parser.add_argument('-norm_vsm', action='store_true', default=False)
parser.add_argument('-norm_neural', action='store_true', default=False)
parser.add_argument('-ensemble', default='vote', help='vote, sum and learn')


opt = parser.parse_args()

if opt.types:
    types_ = opt.types.split(',')
    opt.type_filter = set()
    for t in types_:
        opt.type_filter.add(t)

if opt.test_file.lower() == 'none':
    opt.test_file = None

