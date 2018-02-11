#!/bin/python
import classify
import tarfile
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")

    sentiment.count_vect = CountVectorizer(ngram_range=(1,3))

    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)    
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)

    sentiment.selector = SelectKBest(score_func=chi2,k=args.k)
    sentiment.trainX = sentiment.selector.fit_transform(sentiment.trainX,sentiment.trainy)
    sentiment.devX = sentiment.selector.transform(sentiment.devX)

    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    unlabeled.X = sentiment.selector.transform(unlabeled.X)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def semi_supervised(sentiment,unlabeled_data):
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    train_acc = [classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')]
    dev_acc = [classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')]
    amount = unlabeled_data.X.shape[0]/10
    x = []
    y1 = []
    y2 = []
    it=0
    # while(it<1):
    for i in range(0,unlabeled_data.X.shape[0],amount):
        if i+amount > unlabeled_data.X.shape[0]:
            cur_data = unlabeled_data.data[i:unlabeled_data.X.shape[0]]
        else:
            cur_data = unlabeled_data.data[i:(i+amount)]

        curX = sentiment.count_vect.transform(cur_data)
        curX = sentiment.selector.transform(curX)
        curY = cls.predict(curX)
        curY_prob = cls.predict_proba(curX)
        index = []
        # only expand unlabel data that has probability greater than 0.95
        for n in range(0,curY_prob.shape[0]):
            if args.confidant and (curY_prob[n][0]>=0.95 or curY_prob[n][1]>=0.95):
                index.append(n)
        expand_data = [cur_data[n] for n in index]
        curY = [curY[n] for n in index]
        sentiment.train_data += expand_data
        sentiment.trainy = np.append(sentiment.trainy,np.array(curY))
        sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
        sentiment.trainX = sentiment.selector.fit_transform(sentiment.trainX,sentiment.trainy)
        sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
        sentiment.devX = sentiment.selector.transform(sentiment.devX)
        # weights = cls.coef_[0]
        # print("largest weight "+str(np.amax(weights)))
        # minindex = np.argmin(weights)
        # print("smallest weight "+str(np.amin(weights)))
        # maxindex = np.argmax(weights)

        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
        # print("smallest index weight"+str(cls.coef_[0][minindex]))
        # print("largest index weight"+str(cls.coef_[0][maxindex]))

        print("%8.2f of all unlabeled data included"% float(float(i)/unlabeled_data.X.shape[0]))
        x.append(float(float(i)/unlabeled_data.X.shape[0]))
        devAcc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        y1.append(devAcc)
        y2.append(classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train'))
        if devAcc > 0.7894:
            print("Stop!")
            return cls

    plt.plot(x,y1,'bs',x,y2,'g^')
    plt.show()
    return cls

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write the predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def main(args):
    print(args)
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    old_weights = cls.coef_[0]
    print(cls.coef_[0][:5])

    print("\nSemi-supervised training")
    cls = semi_supervised(sentiment,unlabeled)
    new_weights = cls.coef_[0]
    print(cls.coef_[0][:5])
    dif_weights = old_weights - new_weights
    print("most signifantly diff feature")
    print(np.amax(np.absolute(dif_weights)))
    index = np.argmax(np.absolute(dif_weights))
    print(old_weights[index])
    print(new_weights[index])
    
    print("\nEvaluating again")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    unlabeled = read_unlabeled(tarfname, sentiment)

    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidant',type=bool,default=False)
    parser.add_argument('--token', type=str,default="count")
    parser.add_argument('--select_features',type=bool,default=False)
    parser.add_argument('--k',type=int,default=5000)
    parser.add_argument('--max_df',type=float,default=0.4)
    args = parser.parse_args()
    main(args)