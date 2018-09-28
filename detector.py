import _pickle as c
import os
from sklearn import *
from collections import Counter


def load(clf_file):
    with open(clf_file,'rb') as fp:
        clf = c.load(fp)
    return clf


def make_dict():
    direc = "msg/"
    files = os.listdir(direc)
    message = [direc + msg for msg in files]
    words = []
    c = len(message)
    for msg in message:
        f = open(msg,"r",encoding='utf-8', errors='ignore')
        blob = f.read()
        words += blob.split(" ")
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(3000)


clf = load("text-classifier.mdl")
d = make_dict()


while True:
    features = []
    inp = input(">").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print(["Not Spam", "Spam!"][res[0]])
