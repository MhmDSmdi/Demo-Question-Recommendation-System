import pickle

from hazm import Normalizer, Stemmer, Lemmatizer, sent_tokenize, word_tokenize, stopwords_list

stops = set(stopwords_list())


def load_dataset(file_name, column_name='question'):
    data = pickle.load(open(file_name, "rb"))
    statements = []
    for i in range(len(data)):
        statements.append(data[i][column_name])
    return statements


def statement_pre_processing(input_statement):
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    input_statement = normalizer.normalize(input_statement)
    input_statement = [lemmatizer.lemmatize(word) for word in word_tokenize(input_statement) if word not in stops]
    return input_statement


def dataset_cleaner(dataset):
    statements = []
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    for i in range(len(dataset)):
        normalized_statement = normalizer.normalize(dataset[i])
        # for sentence in sent_tokenize(dataset[i]):
        word_list = [lemmatizer.lemmatize(word) for word in word_tokenize(normalized_statement) if word not in stops]
        statements.append(word_list)
    return statements
