get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from collections import Counter

sms_data = np.loadtxt("C:/Users/TUĞBA KABLAN/Desktop/ML-Spam Clasifier/02-2-SpamClassifier/SMSSpamCollection_cleaned.csv", delimiter="\t", skiprows=1, dtype=str)

checkpoint_data = np.array([['spam', 'dear researcher submit manuscript money'], 
          ['ham','dear friend meet beer'],
          ['ham', 'dear friend meet you']], dtype=str)

sms_data

num_messages = sms_data.shape[0]
print(f"Total message number in data group: {num_messages}")

print("Third messages's etiquet:", sms_data[2][0])

print("Dividing the third message into the words :", sms_data[2][1].split())

def construct_corpus(data):
    """
    np.array[str, str] -> dict[str:int]
    
    from a 2D array of str, return a hash table
    """
    corpus = {}
    row_index = 0  

    for label, message in data:
        words = message.split() 
        
        for word in words:
            if word not in corpus:  
                corpus[word] = row_index  
        
        row_index += 1  

    return corpus  

sms_data = np.array([['ham', 'go until jurong point crazy available only in bugis n great world la e buffet cine there got amore wat'],
                     ['ham', 'ok lar joking wif u oni'],
                     ['spam', 'free entry in 2 a wkly comp to win fa cup final tkts 21st may 2005 text fa to 87121 to receive entry question std txt rate t c s apply 08452810075over18 s']])

corpus = construct_corpus(sms_data)

print({k: corpus[k] for k in list(corpus.keys())[:10]})

def recode_messages(data, corpus):
    """
    np.array[str, str] * dict[str:int] -> np.array[int, int]
    
    returns the binary matrix encoding 
    """
    num_messages = data.shape[0]  
    num_words = len(corpus)  
    matrix = np.zeros((num_messages, num_words)) 
    
    for i, (label, message) in enumerate(data):
        words = message.split()  
        
        for word in words:
            if word in corpus:  
                word_index = corpus[word] 
                matrix[i, word_index] = 1
    
    return matrix

D = construct_corpus(sms_data)  
sms_matrix = recode_messages(sms_data, D)  

print(sms_matrix)


D = construct_corpus(sms_data)

sms_matrix = recode_messages(sms_data, D)

from sklearn.utils import shuffle

def train_test_split(X, Y, train_percentage=0.8):
    assert X.shape[0] == Y.shape[0]

    number_examples = X.shape[0]
    num_train = int(train_percentage * number_examples)
  
    assert X.shape[0] == Y.shape[0]
    
    X, Y = shuffle(X, Y, random_state=42)
    
    number_examples = X.shape[0]
    num_train = int(train_percentage * number_examples)
    
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]
    
    return X_train, Y_train, X_test, Y_test

X = sms_matrix  
Y = sms_data[:, 0] 

X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

print(f"Training set size: {X_train.shape[0]} messages")
print(f"Testing set size: {X_test.shape[0]} messages")
    
import numpy as np

def estimate_proportions(data_matrix, labels):
    """
    Estimate the matrix theta from a binary data matrix and class labels.
    
    """
    d = data_matrix.shape[1]
    theta = np.zeros((d, 2))  

    ham_indices = np.where(labels == 'ham')[0]
    spam_indices = np.where(labels == 'spam')[0]

    N_ham = len(ham_indices)
    N_spam = len(spam_indices)

    for i in range(d):
        n_i_ham = np.sum(data_matrix[ham_indices, i])
        n_i_spam = np.sum(data_matrix[spam_indices, i])

        theta[i, 0] = (n_i_ham + 1) / (N_ham + 2)
        theta[i, 1] = (n_i_spam + 1) / (N_spam + 2)

    return theta

Dico = {'dear': 0, 'researcher': 1, 'money': 2, 'friend': 3}
datam = np.array([
    [1, 0, 1, 1],  
    [1, 1, 0, 1],  
    [0, 0, 1, 1]   
], dtype=int)

labels = np.array(['spam', 'ham', 'ham'])

theta = estimate_proportions(datam, labels)
print(theta)
