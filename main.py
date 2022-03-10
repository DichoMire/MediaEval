from distutils.command import clean
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import logging, codecs, re, matplotlib.pyplot as plt, pandas as pd, nltk

'''
[0]tweetId
[1]tweetText
[2]userId
[3]imageId(s)
[4]username
[5]timestamp
[6]label
'''

def load_raw_dataset_tsv( filename = None ) :

    dataset_raw = []           

    read_handle = codecs.open( filename, 'r', 'utf-8', errors = 'replace' )
    list_lines = read_handle.readlines()
    read_handle.close()

    dataset_raw = []
    for str_line in list_lines :
        split = re.split( r'\t+' , str_line.rstrip('\n'))
        dataset_raw.append( split )

    return dataset_raw

def raw_to_dataframe ( dataset_raw = {} ):

    get_from_tuple = lambda x: x[0]

    dict_dataframe = pd.DataFrame(dataset_raw[1:])
    dict_dataframe.columns=[dataset_raw[0]]
    dict_dataframe.columns = map(get_from_tuple, dict_dataframe.columns)
        
    return dict_dataframe 

def drop_columns (dataframe):
    dataframe.drop(['tweetId','userId', 'imageId(s)', 'username', 'timestamp'], axis=1, inplace=True)
    return dataframe

def float_to_bool (num):
    if(num <= 0): return True
    else: return False

def clean_dataframe (dataframe):

    #=== TRANSFORMING HUMOR INTO FAKE FOR THE LABEL COLUMN
    dataframe['label'] = dataframe['label'].apply(lambda l: re.sub(pattern=r'humor', repl='fake', string=l))

    #=== REMOVING NEW LINE SYMBOLS AND SLASHES ===#
    #dataframe['tweetText'] = dataframe['tweetText'].apply(lambda a: a.replace(to_replace=r'\\n|\\',value='',regex=True))
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda a: re.sub(pattern=r'\\\n|\\',repl='',string=a))
    
    #=== REMOVING WEB LINKS ===#
    #url_filter = lambda a: a.replace()
    # CODE FROM https://stackoverflow.com/questions/51994254/removing-url-from-a-column-in-pandas-dataframe
    # REGEX FROM https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url
    #dataframe['tweetText'] = dataframe['tweetText'].apply(lambda a: a.replace(to_replace=r'((http|https)?\:?(\s)*\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',value='',regex=True))
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda a: re.sub(pattern=r'((http|https)?\:?(\s)*\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',repl='',string=a))

    #=== REMOVING EMOJIS AND UNFORTUNATELY SOME LANGUAGES SUCH AS JP ===#
    #CODE FROM: https://stackoverflow.com/questions/65109065/python-pandas-remove-emojis-from-dataframe
    # DOESNT WORK WITH JAPANESE
    #filter_char = lambda c: ord(c) < 256
    # CURRENTLY WITH LATIN AND GREEK
    filter_char = lambda c: ord(c) < 1024
    # COMMON LANGUAGES IN THE WORLD
    #filter_char = lambda c: ord(c) < 8192

    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda s: ''.join(filter(filter_char, s)))
    #dfList['tweetText'] = dfList['tweetText'].apply(lambda s: [string for string in s if filter(filter_char, s)])
    #dfList['tweetText'] = clear_emojis(dfList['tweetText'])

    #=== REMOVING STOPWORDS ===#
    #list_stopwords = nltk.corpus.stopwords.words('english')
    list_stopwords = nltk.corpus.stopwords.words()
    list_stopwords.extend( [ ':', ';', '[', ']', '"', "'", '(', ')', '.', '?','#', '@'] )

    #CODE TAKEN FROM: https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (list_stopwords)]))

    return dataframe

def lematize_dataframe(dataframe):
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda s: lemmatizeString(s))
    return dataframe

"""
NOT IMPLEMENTED
def decapitalize_letters(dataframe):
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda arr: )
"""

def concat_dataframe(dataframe):
    dataframe['tweetText'] = dataframe['tweetText'].apply(lambda a: ' '.join(a))
    return dataframe

def clear_emojis(stringList):
    # DOESNT WORK WITH JAPANESE, KOREAN, CYRILLIC, etc.
    #filter_char = lambda c: ord(c) < 256
    # CURRENTLY WITH LATIN AND GREEK
    filter_char = lambda c: ord(c) < 1024

    for string in stringList:
        string.filter(filter_char, string)
    return stringList

def tokenizeString(string):
    token = nltk.tokenize.WhitespaceTokenizer()
    return token.tokenize(string)

#CODE TAKEN FROM: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizer
def lemmatizeString(string):
    #Alternatively stemming
    lematizer = nltk.stem.WordNetLemmatizer()
    lematized_array = []
    for word in tokenizeString(string):
        lematized_array.append(lematizer.lemmatize(word))
    return lematized_array

#TRUE IF IS TEST SET
def prepare_raw_dataset(raw, test):
    dfList = raw_to_dataframe(raw)

    dfList = drop_columns(dfList)

    dfList = clean_dataframe(dfList)

    dfList = lematize_dataframe(dfList)

    dfList = concat_dataframe(dfList)

    return dfList

def visualize_score(f1, conf):
    # CODE FROM: https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(x=j, y=i,s=conf[i, j], va='center', ha='center', size='xx-large')

    # CODE FROM: https://stackoverflow.com/questions/45861947/how-to-insert-the-text-below-subplot-in-matplotlib
    ax.text(0.5,-0.13, "F1 score: " + str(f1), size=12, ha="center", 
         transform=ax.transAxes)

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


if __name__ == '__main__' :
    # REQUIRED WHEN RAN FOR THE FIRST TIME !!!!!!!
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    #nltk.download('omw-1.4')
    LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
    logger = logging.getLogger( __name__ )
    logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
    logger.info('logging started')
    strFile = 'mediaeval-2015-trainingset.txt'

    logger.info( 'loading training dataset: ' + strFile )

    listDatasetRaw = load_raw_dataset_tsv( filename = strFile )

    strFile = 'mediaeval-2015-testset.txt'

    logger.info( 'loading training dataset: ' + strFile )

    listDatasetRaw_test = load_raw_dataset_tsv( filename = strFile )

    preparedTraining = prepare_raw_dataset(listDatasetRaw, False)
    #CODE FROM: https://stackoverflow.com/questions/64568775/tf-idf-vectorizer-to-extract-ngrams
    vectorizer = TfidfVectorizer(max_df=0.01, min_df=0.0001, ngram_range=(1,2))
    #matrix = vectorizer.fit_transform(dataframe).todense()
    #return pd.DataFrame(matrix, columns=vectorizer.get_feature_names_out())
    vector_training = vectorizer.fit_transform(preparedTraining['tweetText'])

    counts = pd.DataFrame(vector_training.toarray(),
                      columns=vectorizer.get_feature_names_out())
    # DO NOT UNCOMMENT
    #print(counts.T.sort_values(by=0, ascending=False).head(10))

    preparedTest = prepare_raw_dataset(listDatasetRaw_test, True) 

    vector_test = vectorizer.transform(preparedTest['tweetText'])

    algo = 'mlnb'

    if(algo == 'lnreg'):
        lnreg = LinearRegression()

        preparedTraining['label'] = preparedTraining['label'].apply(lambda l: 0 if (l == 'real') else 1)

        lnreg.fit(vector_training, preparedTraining['label'])

        predicted = lnreg.predict(vector_test)

        #predicted = np.where()predicted.map(float_to_bool, predicted)
        predicted = ['real' if val <=0 else 'fake' for val in predicted]

        """
        for val in range(0, len(predicted)):
            if val <= 0: pred_temp[] = 'real'
            else: val = 'fake'
        logger.info(predicted)
        """

    if(algo == 'mlnb'):
        mlnb = MultinomialNB()

        #preparedTraining[1]['label'] = preparedTraining[1]['label'].apply(lambda l: 0 if (l == 'real') else 1)

        mlnb.fit(vector_training, preparedTraining['label'])

        predicted = mlnb.predict(vector_test)

        """
        for val in predicted:
            if val <= 0: val = 'real'
            else: val = 'fake'
        logger.info(predicted)
        """

    if(algo == 'rndforcla'):
        clf = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state =0)

        clf.fit(vector_training , list(preparedTraining['label']))

        predicted = clf.predict(vector_test)

        #cm = confusion_matrix(prediction, list(test_dataset.label))

        #print("F1 accuracy:")
        #print(f1_score(list(test_dataset['label']),prediction,average = 'micro'))

    conf = confusion_matrix(preparedTest['label'], predicted)

    f1 = f1_score(preparedTest['label'], predicted, average='micro')

    visualize_score(f1, conf)

    logger.info(algo + ' = ' + str(f1))

    #logger.info(conf)
    print(conf)

    #end
