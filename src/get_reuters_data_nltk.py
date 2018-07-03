from os.path import expanduser
from collections import defaultdict
from nltk.corpus import reuters
import data as dt
import nltk
import numpy as np

print(nltk.data.path)
home = expanduser("~")
id2cat = defaultdict(list)
path = nltk.data.path[len(nltk.data.path)-1]+'/corpora/reuters/'
cat_names = defaultdict(int)
i = 0
for line in open(path + 'cats.txt','r'):
    fid, _, cats = line.partition(' ')
    for c in cats.split():
        if c not in cat_names:
            cat_names[c] = i
            print("found", cats, i)
            i += 1
    id2cat[fid] = cats.split()

# SAVE THE CAT MAPPING

docs = []
fileid_mapping = defaultdict(int)
i = 0
for fileid in reuters.fileids():
    doc = dt.import1dArray(path + fileid)
    doc = " ".join(doc)
    docs.append(doc)
    fileid_mapping[fileid] = i
    print(doc)
    i += 1

class_all = np.zeros(shape=(len(docs), len(cat_names.keys())), dtype=np.int8)
new_class_all = np.zeros(shape=(len(docs), len(cat_names.keys())), dtype=np.int8)
i = 0
for line in open(path + 'cats.txt','r'):
    fid, _, cats = line.partition(' ')
    doc_index = fileid_mapping[fid]
    for c in cats.split():
        class_index = cat_names[c]
        class_all[doc_index][class_index] = 1
        new_class_all[doc_index][class_index] = 1
        print(fid, doc_index, c, class_index)

print(class_all.shape)



save_path = "../data/raw/reuters/"

np.save(save_path + "fileid_mapping.npy", fileid_mapping)
np.save(save_path + "category_name_mapping.npy", cat_names)
print("cats", len(np.unique(list(cat_names.keys()))))
dt.write1dArray(list(cat_names.keys()), save_path + "category_names.txt")

names = list(fileid_mapping.keys())
for i in range(len(names)):
    names[i] = "_".join(names[i].split("/"))

dt.write1dArray(names, save_path + "available_entities.txt")
print("names", len(np.unique(names)))

dt.write2dArray(class_all, save_path + "class-all.txt")
dt.write1dArray(docs, save_path + "corpus.txt")

print("docs", len(np.unique(docs)))

unique_docs, index = np.unique(docs, return_index=True)

copies = []
for i in range(len(docs)):
    if i not in index:
        copies.append(i)

found_copies = []
all_copy_indexes = []
for i in copies:
    copy_index = [i]
    found_copies.append(i)
    print("")
    print(i, docs[i])
    for j in range(len(docs)):
        if docs[j] == docs[i] and i != j and j not in found_copies:
            copy_index.append(j)
            found_copies.append(j)
            print(j, docs[j])
    print(copy_index)
    all_copy_indexes.append(copy_index)

inds_to_del = []

for i in range(len(all_copy_indexes)):
    if len(all_copy_indexes[i]) == 1:
        inds_to_del.append(i)

all_copy_indexes = np.delete(all_copy_indexes, inds_to_del, axis=0)

print(len(all_copy_indexes))
"""
different_class_index = []
for i in range(len(all_copy_indexes)):
    found = False
    for j in range(len(all_copy_indexes[i])):
        for x in range(len(class_all[all_copy_indexes[i][j]])):
            if class_all[all_copy_indexes[i][j]][x] != class_all[all_copy_indexes[i][0]][x]:
                print("different classes", class_all[all_copy_indexes[i]])
                print("for", docs[all_copy_indexes[i][0]])
                different_class_index.append(all_copy_indexes[i])
                found = True
                break
        if found is True:
            break

print(len(different_class_index))
"""
# For each first instance of the duplicates, change their classes to fit all of their duplicates



final_inds = []

for i in range(len(all_copy_indexes)):
    first_try = False
    final_inds.append(all_copy_indexes[i][1])
    for j in range(len(all_copy_indexes[i])):
        for x in range(len(class_all[all_copy_indexes[i][j]])):
            if class_all[all_copy_indexes[i][j]][x] == 1 and class_all[all_copy_indexes[i][1]][x] != 1:
                new_class_all[all_copy_indexes[i][1]][x] = 1
                if not first_try:
                    print(i, j, all_copy_indexes[i][j], x)
                first_try = True

docs = unique_docs

for i in final_inds:
    if i not in index:
        print(i, "not in index")
unique_class_all = new_class_all[index]
old_u_class_all = class_all[index]
diffcount = 0
for i in range(len(old_u_class_all)):
    if np.array_equal(old_u_class_all[i], unique_class_all[i]) is False:
        print("diff", i, old_u_class_all[i], unique_class_all[i])
        diffcount += 1
print(diffcount)

names = np.asarray(names)[index]
sorted_inds = np.flipud(np.argsort(names))

names = names[sorted_inds]
docs = docs[sorted_inds]
unique_class_all = unique_class_all[sorted_inds]
index = index[sorted_inds]




unique_class_all, class_names = dt.removeInfrequent(unique_class_all, list(cat_names.keys()))

dt.write2dArray(unique_class_all, save_path + "duplicate_removed_classes.txt")
dt.write1dArray(class_names, save_path + "category_names_20freq_removed.txt")
dt.write1dArray(docs, save_path + "duplicate_removed_docs.txt")
dt.write1dArray(index, save_path + "unique_doc_index_from_orig.txt")
dt.write1dArray(names, save_path + "duplicate_removed_available_entities.txt")

# Delete all duplicates