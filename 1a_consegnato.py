import random
import abc
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree, svm
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


try:
    from xgboost import XGBClassifier
except:
    print('WARNING: XGBoost Classifier is not available due to missing dependency')


def load_data(deduplicate = False, split = 0.85):
    sentences = []
    labels = []

    with open('C:\\Users\\Martino Fabiani\\OneDrive\\Documenti\\Workspace\\dialog_acts.dat', 'r') as file_handle:
        lines = [line.rstrip() for line in file_handle]
    
    # randomize the order so that the last 15% of entries represent
    # a random sample of the overall dataset
    random.seed(27)
    random.shuffle(lines)

    for line in lines:
        idx = line.find(' ')
        
        label = line[:idx].strip().lower()
        sentence = line[idx:].strip().lower()
        
        if deduplicate and sentence in sentences:
            continue
        else:
            labels.append(label)
            sentences.append(sentence)

    split_idx = int(split * len(sentences))
    
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    
    return (train_sentences, train_labels), (test_sentences, test_labels)


# abstract class which provides a template to create different classifiers
class Classifier(abc.ABC):
    @abc.abstractmethod
    def train(self, sentences, labels):
        pass
    
    @abc.abstractmethod
    def predict(self, sentence):
        pass
    
    def batch_predict(self, sentences):
        vectorized_predict_function = np.vectorize(self.predict)
        predictions = vectorized_predict_function(sentences)
        return predictions


class MajorityClassClassifier(Classifier):
    
    def train(self, sentences, labels):
        # find the label which is present most often in the training data
        self.majority_class = max(set(labels), key=labels.count)
        
        # the class should be equal to the 'inform' class according to the documentation
        assert self.majority_class == 'inform'
    
    def predict(self, sentence):
        return self.majority_class


class RuleBasedClassifier(Classifier):
    
    def train(self, sentences, labels):
        self.default = 'inform'
        self.rules = [
            (lambda sentence: 'kay ' in sentence, 'ack'),
            (lambda sentence: 'yes' in sentence, 'affirm'),
            (lambda sentence: 'goodbye' in sentence, 'bye'),
            (lambda sentence: 'is it' in sentence, 'confirm'),
            (lambda sentence: 'does it' in sentence, 'confirm'),
            (lambda sentence: 'do they' in sentence, 'confirm'),
            (lambda sentence: 'i dont want' in sentence, 'deny'),
            (lambda sentence: 'hello' in sentence, 'hello'),
            (lambda sentence: 'hi i am' in sentence, 'hello'),
            (lambda sentence: 'looking for' in sentence, 'inform'),
            (lambda sentence: 'no' in sentence, 'negate'),
            (lambda sentence: 'again' in sentence, 'repeat'),
            (lambda sentence: 'go back' in sentence, 'repeat'),
            (lambda sentence: 'anything else' in sentence, 'reqalts'),
            (lambda sentence: 'how about' in sentence, 'reqalts'),
            (lambda sentence: 'more' in sentence, 'reqmore'),
            (lambda sentence: 'phone number' in sentence, 'request'),
            (lambda sentence: 'address' in sentence, 'request'),
            (lambda sentence: 'start over' in sentence, 'restart'),
            (lambda sentence: 'thank you' in sentence, 'thankyou'),
            (lambda sentence: 'noise' in sentence, 'null'),
            (lambda sentence: 'cough' in sentence, 'null'),
            (lambda sentence: 'unintelligible' in sentence, 'null'),
        ]

    def predict(self, sentence):
        
        for rule in self.rules:
            matches_rule, label = rule
            
            if matches_rule(sentence):
                return label
        
        # if none of the rules match
        return self.default

class NeuralNetworkClassifier(Classifier):

    def train(self, sentences, labels):
        # the labels given by the dataset are strings, first convert each label
        # to a corresponding integer, then convert each integer to a binary class vector
        # which can be used to train the model
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        num_classes = len(self.label_encoder.classes_)
        
        y_train = self.label_encoder.transform(labels)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        
        # the list of training sentences needs to be a numpy array
        x_train = np.array(sentences)
        
        # use the 'binary' vectorization mode to build a bag-of-words model
        text_vectorization_layer = keras.layers.TextVectorization(output_mode='binary')
        text_vectorization_layer.adapt(x_train)

        self.model = keras.Sequential()
        self.model.add(text_vectorization_layer)
        self.model.add(keras.layers.Dense(num_classes, activation='softmax'))
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = 'accuracy')

        self.model.fit(
            x_train,
            y_train,
            epochs = 12,
            verbose = 1,
            validation_split = 0.2,
            shuffle = True)
        
    def predict(self, sentence):
        prediction = self.model.predict([sentence], verbose = 0)
        encoded_prediction = np.argmax(prediction, axis = 1)
        labeled_prediction = self.label_encoder.inverse_transform(encoded_prediction)
        return labeled_prediction
    
    def batch_predict(self, sentences):
        predictions = self.model.predict(sentences, verbose = 0)
        encoded_predictions = np.argmax(predictions, axis = 1)
        labeled_predictions = self.label_encoder.inverse_transform(encoded_predictions)
        return labeled_predictions
        
class XGBoostClassifier(Classifier):

    def train(self, sentences, labels):
        # the labels given by the dataset are strings, first convert each label
        # to a corresponding integer, then convert each integer to a binary class vector
        # which can be used to train the model
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        num_classes = len(self.label_encoder.classes_)
        
        x_train = np.array(sentences)
        y_train = self.label_encoder.transform(labels)
        
        # use the 'binary' vectorization mode to build a bag-of-words model
        self.text_vectorization_layer = keras.layers.TextVectorization(output_mode='binary')
        self.text_vectorization_layer.adapt(x_train)

        self.model = XGBClassifier(num_class=num_classes)
        x_train = np.array(x_train, dtype='object')
        x_train = self.text_vectorization_layer(x_train)
        self.model.fit(x_train, y_train)
        
    def predict(self, sentence):
        sentence = self.text_vectorization_layer(sentence)
        prediction = self.model.predict([sentence])
        labeled_prediction = self.label_encoder.inverse_transform(prediction)
        return labeled_prediction
    
    def batch_predict(self, sentences):
        sentences = self.text_vectorization_layer(sentences)
        predictions = self.model.predict(sentences)
        labeled_predictions = self.label_encoder.inverse_transform(predictions)
        return labeled_predictions


class DecisionTreeClassifier(Classifier):

    def train(self, sentences, labels):
        # Vectorize sentences
        self.tfidf_vectorizer = TfidfVectorizer()
        x_train = self.tfidf_vectorizer.fit_transform(sentences)
        
        self.DTC = tree.DecisionTreeClassifier()
        self.DTC.fit(x_train, labels)
    
    def predict(self, sentence):
        sentence = self.tfidf_vectorizer.transform([sentence])
        prediction = self.DTC.predict(sentence)[0]
        return prediction
    
    def batch_predict(self, sentences):
        sentences = self.tfidf_vectorizer.transform(sentences)
        predictions = self.DTC.predict(sentences)
        return predictions


class SupportVectorClassifier(Classifier):

    def train(self, sentences, labels):
        # Vectorize sentences
        self.tfidf_vectorizer = TfidfVectorizer()
        x_train = self.tfidf_vectorizer.fit_transform(sentences)
        
        self.SVC = svm.SVC()
        self.SVC.fit(x_train, labels)

    def predict(self, sentence):
        sentence = self.tfidf_vectorizer.transform([sentence])
        prediction = self.SVC.predict(sentence)[0]
        return prediction
       
    def batch_predict(self, sentences):
        sentences = self.tfidf_vectorizer.transform(sentences)
        predictions = self.SVC.predict(sentences)
        return predictions
        

def evaluate(classifier: Classifier, test_sentences, test_labels):
    predictions = classifier.batch_predict(test_sentences)

    accuracy = accuracy_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division = 0.0)  
    precision = precision_score(test_labels, predictions, average='weighted', zero_division = 0.0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division = 0.0)
    confusion = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    print('Classifier Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1 Score: {:.3f}'.format(f1))
    print('Report: ' + report)
    print('Confusion Matrix:')
    # make sure that each row of the confusion matrix is not slit over multiple lines
    # when printing to the standard output
    
     # Stampa la matrice di confusione in modo dettagliato
    labels = np.unique(test_labels)
    header = 'Predicted →'
    print('{:20}'.format(header), end='')
    for label in labels:
        print('{:10}'.format(label), end='')
    print()

    for i, row_label in enumerate(labels):
        print('{:20}'.format(row_label), end='')
        for j in range(len(labels)):
            print('{:10}'.format(confusion[i, j]), end='')
        print()

    labels = np.unique(test_labels)
    confusion = confusion_matrix(test_labels, predictions)

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cm ap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Aggiungi i conteggi all'interno dei quadrati
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(confusion[i, j]), horizontalalignment='center', color='black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
    


def main():
    print('What classifier would you like to use?')
    print('1. Baseline: Majority')
    print('2. Baseline: Rule-Based')
    print('3. Machine Learning: Neural Network Classifier')
    print('4. Machine Learning: XGBoost Classifier')
    print('5. Machine Learning: Decision Tree Classifier')
    print('6. Machine Learning: C-Support Vector Classifier')
    
    classifier_id = input('[1-6]: ')
    if classifier_id == '1':
        classifier = MajorityClassClassifier()
    elif classifier_id == '2':
        classifier = RuleBasedClassifier()
    elif classifier_id == '3':
        classifier = NeuralNetworkClassifier()
    elif classifier_id == '4':
        classifier = XGBoostClassifier()
    elif classifier_id == '5':
        classifier = DecisionTreeClassifier()
    elif classifier_id == '6':
        classifier = SupportVectorClassifier()
    else:
        classifier = MajorityClassClassifier()

        print('Input not selected correctly, defaulting to Majority Classifier')
    
    
    print('Do you want to deduplicate the dataset?')
    dedup = input('[y/n]: ').lower()
    if dedup in {'y', 'n'}:
        dedup = True if dedup == 'y' else False
    else:
        dedup = False
        print('Input not selected correctly, defaulting to no deduplication')


    (train_sentences, train_labels), (test_sentences, test_labels) = load_data(dedup)

    classifier.train(train_sentences, train_labels)
    evaluate(classifier, test_sentences, test_labels)
    

    print('Use either .exit or .quit to exit Chat Mode.')
    while True:
        sentence = input('Please enter an utterance: ').lower()
        
        if sentence == '.exit' or sentence == '.quit':
            break
        
        label_prediction = classifier.predict(sentence)
        print('classifier prediction: {0}'.format(label_prediction))


if __name__ == '__main__':
    main()
