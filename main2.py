import sklearn
import nltk
import csv
import re
from nltk.stem.porter import PorterStemmer
import sys
import numpy as np
import os.path
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import init
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn.metrics

stemmer = PorterStemmer()

def main(test_model_locally):
    init()

    ans = ""
    # get the dataset
    print colored("Please provide train file:", 'cyan', attrs=['bold'])
    ans = sys.stdin.readline()
    # remove any newlines or spaces at the end of the input
    path = ans.strip('\n').rstrip(' ')
    file_train = path or 'dataset/train/train.dat'

    print colored("Please provide test file:", 'cyan', attrs=['bold'])
    ans = sys.stdin.readline()
    # remove any newlines or spaces at the end of the input
    path = ans.strip('\n').rstrip(' ')
    file_test = path or 'dataset/test/test.dat'

    # test purpose only
    file_format = 'dataset/test/format.dat'

    print '\n'

    # 1. digestive system diseases,
    # 2. cardiovascular diseases,
    # 3. neoplasms,
    # 4. nervous system diseases,
    # 5. general pathological conditions
    categories = [1, 2, 3, 4, 5]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    dataList = []
    categoryList = []
    testDataList = []
    test_labels = []

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])

    # load train data

    with open(file_train) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            category = row[0]
            doc = normalize(row[1], stop_words)
            categoryList.append(category)
            dataList.append(doc)

    # load test data
    with open(file_test) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            testDataList.append(normalize(row[0], stop_words))

    # #load format/test data labels
    # with open(file_format) as f:
    #     reader = csv.reader(f, delimiter='\t')
    #     for row in reader:
    #         test_labels.append(row[0])

    # do the main test
    main_test(dataList, categoryList, testDataList, test_labels, stop_words, test_model_locally)


def main_test(train_data, train_label, test_data, test_label, stop_words, test_model_locally):
    # Extracts features for given text: train or test
    train_x, test_x = extract_features(train_data, train_label, test_data, stop_words)
    train_y = train_label

    # create classifier
    clf = LinearSVC()

    print '\n'

    if test_model_locally:
        test_model_80_20(clf, train_x, train_y)
    else:
        # train the model without split
        clf.fit(train_x, train_y)
        test_prediction = clf.predict(test_x)
        #print sklearn.metrics.classification_report(test_label, test_prediction, digits=2)

        # Print accuracy
        # accuracy = np.sum(test_prediction == test_label).astype(float) / len(test_label)
        # print("Accuracy: " + str(accuracy * 100) + '%')

        print "\n".join(test_prediction)


def test_model_80_20(clf, train_x, train_y):
    # test the classifier
    print colored('Testing classifier with test data', 'magenta', attrs=['bold'])

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    #print sklearn.metrics.classification_report(y_test, prediction, digits=2)
    accuracy = np.sum(prediction == y_test).astype(float) / len(y_test)
    print("Accuracy: " + str(accuracy * 100) + '%')


def extract_features(train_data, train_label, test_data, stop_words):
    # TF-IDF with n-gram features
    # TFIDF features: Term Frequency Inverse Document Frequency
    vectorizer = TfidfVectorizer()

    train_x_fit = vectorizer.fit_transform(train_data).toarray()
    # Only transforming test data, not fitting it
    test_x_fit = vectorizer.transform(test_data).toarray()

    print train_x_fit.shape, test_x_fit.shape, train_x_fit.dtype, test_x_fit.dtype
    return train_x_fit, test_x_fit


def normalize(text,stop_word):
    #print text

    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    words = text.split(' ')

    # Remove stop words
    words = [word for word in words if word not in stop_word]

    # Stem the data
    stemmed = []
    for item in words:
        stemmed.append(stemmer.stem(item))

    text = " ".join(stemmed) + " ".join(words)

    #print text

    return text


if __name__ == '__main__':
    # set to true only when testing locally. Does an 80/20 split on train data.
    # true will not predict on test data.
    test_model_locally = False
    main(test_model_locally)
