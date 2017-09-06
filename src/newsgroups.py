from sklearn.datasets import fetch_20newsgroups
import data as dt
import numpy as np



# Import the newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=False)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=False)

train_len = len(newsgroups_train.data)
test_len = len(newsgroups_test.data)

print(newsgroups_train.target[train_len-1])
print(newsgroups_train.target[train_len-2])
print(newsgroups_train.target[train_len-3])
print(newsgroups_test.target[0])
print(newsgroups_test.target[1])
print(newsgroups_test.target[2])


vectors = np.concatenate((newsgroups_train.data, newsgroups_test.data), axis=0)
classes = np.concatenate((newsgroups_train.target, newsgroups_test.target), axis=0)

print(classes[train_len-1])
print(classes[train_len-2])
print(classes[train_len-3])
print(classes[train_len])
print(classes[train_len+1])
print(classes[train_len+2])

# Convert them to vectors





# Convert vectors to our format




# Convert classes to our format




