import gensim
# for loading a stored model 
from gensim.models import KeyedVectors

# pandas is a useful package for dealing with data structures
import pandas as pd

#loading a stored model. 

# Please make sure that the path `../models/GoogleNews-vectors-negative300.bin.gz' points to the location where you stored your word embeddings 
# if you are using a non-binary model, you will need to change binary=True to binary=False
ds_model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)

# a first test with the model (you can replace "student" by other words)
ds_model.most_similar("student")
# similarity: a small scale experiment. Feel free to play with this and replace the terms

cos_man_woman = ds_model.similarity('man', 'woman')
cos_man_dog = ds_model.similarity('man', 'dog')


print(f'Man and woman should be more similar than man and dog:')
if cos_man_woman > cos_man_dog:
    print('True!')
    print('man-woman', cos_man_woman)
    print('man-dog', cos_man_dog)
else:
    print('False')
    print('man-woman', cos_man_woman)
    print('man-dog', cos_man_dog)


simlex_data = pd.read_csv('../SimLex-999/SimLex-999.txt',sep='\t')
print(simlex_data)


ds_scores = {}
human_scores = []
model_scores = []

for index, row in simlex_data.sort_values(by='SimLex999', ascending=False).iterrows():
    wordpair = row['word1'] + '-' + row['word2']
    human_scores.append(row['SimLex999'])
    ds_score = ds_model.similarity(row['word1'],row['word2'])
    model_scores.append(ds_score)
    ds_scores[wordpair] = ds_model.similarity(row['word1'],row['word2'])

    
### Also saving the ranked output by the model to a file for inspection
ds_ranked_output = open('../ds_output_simlex_pairs.txt', 'w')
for index, word_pair in enumerate(sorted(ds_scores, key=ds_scores.get, reverse=True)):
    ds_ranked_output.write(str(index) + '\t' + word_pair + '\t' + str(ds_scores[word_pair]) + '\n')
    

#calculate spearman rho

from scipy.stats import spearmanr

out = spearmanr(human_scores, model_scores)
print(out)

    