import nltk
import pickle
import numpy
from collections import Counter
import sys
import json
#sys.path.append('../cocoapi/PythonAPI')
#from pycocotools.coco import COCO
nltk.download('punkt')
args={
    'caption_path':'../json_captions/json_captions.json',                #path for train annotation file
    'vocab_path':'../data/vocab.pkl',                                            #path for saving vocabulary wrapper
    'threshold': 4                                                                                            #minimum word count threshold
}

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_file, threshold):
    """Build a simple vocabulary wrapper."""
    #coco = COCO(json)
    counter = Counter()
    f = open(json_file)
    data = json.load(f)
    #print(type(data["captions"]))
    print(data.keys())
    ids = data['captions'].keys()
    for i, id in enumerate(ids):
        caption = str(data['captions'][id])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    vocab = build_vocab(json_file=args['caption_path'],
                        threshold=args['threshold'])
    print(vocab)
    vocab_path = args['vocab_path']
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    main()
