import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import _pickle as c


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print ("saved")

def read_csv():
    import csv
    f = open('spam.csv','rb')
    reader = csv.reader(f)
    coun = 0
    for i in reader:
        if i[0] == "ham":
            f = open('./nospam/ham'+str(coun),'w+')
            f.write(i[1])
            
        else:
            f = open('./spam/spam'+str(coun),'w+')
            f.write(i[1])
        coun += 1

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


def make_dataset(dictionary):
    direc = "msg/"
    files = os.listdir(direc)
    message = [direc + msg for msg in files]
    feature_set = []
    labels = []
    c = len(message)

    for msg in message:
        data = []
        f = open(msg,"r",encoding='utf-8', errors='ignore')
        words = f.read().split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)

        if "ham" in msg:
            labels.append(0)
        if "spam" in msg:
            labels.append(1)
        c = c - 1 
    return feature_set, labels


d = make_dict()
features, labels = make_dataset(d)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

clf = MultinomialNB()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print (accuracy_score(y_test, preds))
save(clf, "text-classifier.mdl")