import multiprocessing
import gensim

from gensim.models import Phrases, Word2Vec, FastText
from gensim.models.phrases import Phraser
from gensim.similarities import WmdSimilarity
from gensim.test.utils import datapath
from gensim.test.utils import common_texts

from MedicalQuestionSimilarity import data_handler


def load_pre_trained_model(file_name, encoding='utf-8'):
    # model = gensim.models.KeyedVectors.load_word2vec_format(file_name)
    model = FastText.load_fasttext_format(file_name)
    # model.save('fasttext_fa_model')
    return model


def train_word2vec_bigram(word_statements, name='word2vec_fa_model'):
    phrases = Phrases(word_statements, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[word_statements]
    num_cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=20,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=num_cores - 1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save(name)
    w2v_model.init_sims(replace=True)
    return w2v_model


def progbar(curr, full_progbar):
    frac = curr / full_progbar
    filled_progbar = round(frac * full_progbar)
    print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')


progbar(0, 100)
model = load_pre_trained_model('../data/fa.bin')
# model = load_pre_trained_model('../data/cc.fa.300.vec')
progbar(45, 100)

medical_questions = data_handler.load_dataset("../data/data_all.pickle")
medical_questions_words = data_handler.dataset_cleaner(medical_questions)
progbar(55, 100)

# model.build_vocab(medical_questions_words, update=True)
model.train(common_texts, total_examples=len(model.corpus_count), epochs=model.epochs)
progbar(70, 100)
instance = WmdSimilarity(medical_questions_words, model, num_best=10)

user_question = ['آیا برای عمل لیزیک باید ناشتا بود؟',
                 'برای عمل لیزیک نباید سیگار کشید؟',
                 'سلام خسته نباشید چشمم درد میکنه خواستم بدونم باید چیکار کنم؟',
                 'فوتبال ورزش پر هیجانی است']

for i in range(len(user_question)):
    query = data_handler.statement_pre_processing(user_question[i])
    sims = instance[query]
    print('Query: ' + user_question[i])
    for j in range(10):
        print(medical_questions[sims[j][0]] + "("+'sim = %.4f' % sims[j][1]+")")
    print()
    progbar(i * 5 + 75, 100)
progbar(100, 100)
