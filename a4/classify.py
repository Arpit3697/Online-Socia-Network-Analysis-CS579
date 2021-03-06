from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import re

def download_afin():

    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    file = ZipFile(BytesIO(url.read()))
    afin_f = file.open('AFINN/AFINN-111.txt')
    return afin_f

def read_data(file):

    afinn = {}
    for l in file:
        line = l.strip().split()
        if len(line) == 2:
            afinn[line[0].decode("utf-8")] = int(line[1])
    return afinn


def afinn_sentiment2(terms, afinn, verbose=False):

    positive = 0
    negative = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                positive += afinn[t]
            else:
                negative += -1 * afinn[t]
    return positive, negative

def tokenize(text):

    token = re.sub('\W+', ' ', text.lower()).split()
    return token

def token_features(tokens, feats):

    prepend = "token="
    count = Counter(tokens)
    for t in tokens:
        feats[prepend + t]=count[t]
    pass


def token_pair_features(tokens, feats, k=3):

    count_tokens_pairs = Counter()
    for t in range(0,len(tokens)+1):
        if ((t + k <= len(tokens))):
            count_tokens_pairs.update(list(combinations(tokens[t:t + k], 2)))

    for k in count_tokens_pairs:
        feats["token_pair=" + k[0] + "__" + k[1]] = count_tokens_pairs[k]
    pass


def lexicon_features(tokens, feats):

    feats['pos_words'] = 0
    feats['neg_words'] = 0

    for t in tokens:
        if(t.lower() in pos_words):
            feats['pos_words']+=1
        if(t.lower() in neg_words):
            feats['neg_words']+=1
    pass


def featurize(tokens, feature_fns):

    feats = defaultdict(lambda:0)

    for feat_func in feature_fns:
        feat_func(tokens,feats)

    return sorted(feats.items(), key=lambda x:x[0])

    pass

def pos_neg(tweets,tokens,afin):

    postive = []
    negative = []
    combo = []
    for tk, tw in zip(tokens, tweets):
        pos, neg = afinn_sentiment2(tk, afin)
        if neg == pos:
            combo.append((tw['text'], pos, neg))
        elif neg > pos:
            negative.append((tw['text'], pos, neg))
        elif pos > neg:
            postive.append((tw['text'], pos, neg))
    return postive, negative, combo

def main():

    tweets = pickle.load(open('tweets.pickle', 'rb'))
    afin = download_afin()
    read = read_data(afin)
    tokens = [tokenize(t['text']) for t in tweets]
    positives, negatives, combined =pos_neg(tweets,tokens,read)
    print("length of positive, negative, combined is:")
    print(len(positives),len(negatives),len(combined))
    pickle.dump((positives,negatives,combined), open('classify.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
