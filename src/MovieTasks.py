
import data as dt
import re
import numpy as np
import string
from collections import defaultdict
import random
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import pandas as pd
import scipy.sparse as sp
def  getVectors(input_folder, file_names_fn, extension, output_folder, only_words_in_x_entities,
               words_without_x_entities, cut_first_line=False, get_all=False, additional_name="", make_individual=True,
               classification="", use_all_files="", minimum_words=0, data_type="", sparse_matrix=False, word_count_amt = 0):
    if use_all_files is None:
        file_names = dt.import1dArray(file_names_fn)
    else:
        file_names = dt.getFns(use_all_files)

    phrase_dict = defaultdict(int)
    failed_indexes = []
    failed_filenames = []
    working_filenames = []

    # First, get all possible phrase names and build a dictionary of them from the files

    for f in range(len(file_names)):
        try:
            full_name = input_folder + file_names[f] + "." + extension
            phrase_list = dt.import2dArray(full_name, "s")
            if cut_first_line:
                phrase_list = phrase_list[1:]
            word_count = 0
            for p in phrase_list:
                word_count += int(p[1])
            if word_count > word_count_amt:
                for p in phrase_list:
                    if p[0] != "all":
                        phrase_dict[p[0]] += 1
                    else:
                        print("found class all")
                working_filenames.append(file_names[f])
            else:
                print("Failed, <1k words", file_names[f], f, word_count)
                failed_filenames.append(file_names[f])
                failed_indexes.append(f)
        except FileNotFoundError:
            print("Failed to find", file_names[f], f)
            failed_filenames.append(file_names[f])
            failed_indexes.append(f)
    print(failed_indexes)
    print(failed_filenames)
    phrase_sets = []
    # Convert to array so we can sort it
    phrase_list = []


    entity_names = dt.import1dArray(file_names_fn)
    matching_filenames = []
    failed_fns = []
    if data_type == "wines":
        for e in entity_names:
            found = False
            for f in working_filenames:

                if "zz" in f:
                    new_f = f[2:]
                else:
                    new_f = f
                if dt.removeEverythingFromString(e) == dt.removeEverythingFromString(new_f):
                    matching_filenames.append(f)
                    found = True
                    break
            if not found:
                failed_fns.append(e)

        working_filenames = np.unique(np.asarray(matching_filenames))

    test_dupes = np.unique(np.asarray(working_filenames))
    print(len(test_dupes))

    for key, value in phrase_dict.items():
        if value >= only_words_in_x_entities:
            phrase_list.append(key)
    all_phrases = []
    for key, value in phrase_dict.items():
        all_phrases.append(key)

    phrase_sets.append(phrase_list)
    counter = 0
    for phrase_list in phrase_sets:
        if not get_all and counter > 0:
            break
        all_phrase_fn = output_folder+"frequency/phrases/" + "class-all-" +str(only_words_in_x_entities) + "-"+str(words_without_x_entities)+"-"+ classification
        phrase_name_fn = output_folder + "names/"  +str(only_words_in_x_entities) + "-"+str(words_without_x_entities)+"-"+ classification +".txt"
        phrase_list = sorted(phrase_list)

        print("Found", len(phrase_list), "Phrases")
        print(phrase_list[:20])
        print("Failed", len(failed_filenames), "Files")
        print(failed_filenames[:20])

        phrase_index_dict = defaultdict()

        # Create a dictionary to obtain the index of a phrase that's being checked

        for p in range(len(phrase_list)):
            phrase_index_dict[phrase_list[p]] = p

        # Create an empty 2d array to store a matrix of movies and phrases
        all_phrases_complete = []
        for f in working_filenames:
            all_phrases_complete.append([0]*len(phrase_list))

        all_phrases_complete = np.asarray(all_phrases_complete)

        print("Each entity is length", len(all_phrases_complete[0]))
        print("The overall matrix is", len(all_phrases_complete))
        if sparse_matrix:
            all_phrases_complete = sp.csr_matrix(all_phrases_complete)


        # Then, populate the overall bag of words for each film (with all other phrases already set to 0

        completed_index = []

        if data_type == "wines":

            print("wines")
            """
            merge_indexes = []
            for f in range(len(working_filenames)):
                print(working_filenames[f])
                for i in range(len(working_filenames)):
                    if i == f:
                        continue
                    for ci in completed_index:
                        if i == ci:
                            continue
                    if "~" in working_filenames[i]:
                        if working_filenames[f] == working_filenames[i][:-1] or working_filenames[f] == working_filenames[i][2:-1]:
                            completed_index.append(i)
                            merge_indexes.append((f, i))
            """

        for f in range(len(working_filenames)):
            n_phrase_list = dt.import2dArray(input_folder + working_filenames[f] + "." + extension, "s")
            if cut_first_line:
                n_phrase_list = n_phrase_list[1:]
            for p in n_phrase_list:
                phrase = p[0]
                try:
                    phrase_index = phrase_index_dict[phrase]
                    if not sparse_matrix:
                        all_phrases_complete[f][phrase_index] = int(p[1])
                    else:
                        all_phrases_complete[f, phrase_index] = int(p[1])

                    #print("Kept", phrase)
                except KeyError:
                    continue
                    #print("Deleted phrase", phrase)
        """

        cols_to_delete = []
        if data_type == "wines":
            for mt in merge_indexes:
                for v in range(len(all_phrases_complete)):
                    all_phrases_complete[v][mt[0]] += all_phrases_complete[v][mt[1]]
                cols_to_delete.append(mt[1])
        all_phrases_complete = np.delete(all_phrases_complete, cols_to_delete, 1)
        working_filenames = np.delete(working_filenames, cols_to_delete)
        """

        # Import entities specific to the thing
        # Trim the phrases of entities that aren't included in the classfication
        if classification != "all" and classification != "mixed" and classification != "genres" and classification != "ratings" and classification != "types":
            classification_entities = dt.import1dArray("../data/" + data_type + "/classify/" + classification + "/available_entities.txt")
            all_phrases_complete = dt.match_entities(all_phrases_complete, classification_entities, file_names)
        elif classification == "all":
            print("All~~~~~~~~~~~~~~")
            dt.write1dArray(working_filenames, "../data/"+data_type+"/classify/"+classification+"/available_entities.txt")
        if not sparse_matrix:
            all_phrases_complete = np.asarray(all_phrases_complete).transpose()
        else:
            all_phrases_complete = all_phrases_complete.transpose()

        indexes_to_delete = []
        if sparse_matrix:
            cx = sp.coo_matrix(all_phrases_complete)

            indexes_to_delete = []

            for i, j, v in zip(cx.row, cx.col, cx.data):
                print
                "(%d, %d), %s" % (i, j, v)
        for a in range(len(all_phrases_complete)):
            if np.count_nonzero(all_phrases_complete[a]) > len(all_phrases_complete[a]) - (words_without_x_entities):
                print("Recorded an entity " + str(phrase_list[a]) + " with too little difference")
                indexes_to_delete.append(a)
        indexes_to_delete.sort()
        indexes_to_delete.reverse()
        for i in indexes_to_delete:
            all_phrases_complete = np.delete(all_phrases_complete, i, 0)
            print("Deleted an entity " + str(phrase_list[i]) + " with too little difference")
            phrase_list = np.delete(phrase_list, i, 0)

        dt.write1dArray(phrase_list, phrase_name_fn)
        if make_individual:
            for p in range(len(all_phrases_complete)):
                dt.write1dArray(all_phrases_complete[p], output_folder+"frequency/phrases/class-" + phrase_list[p] +
                                 "-"+str(only_words_in_x_entities) + "-"+str(words_without_x_entities)+"-"+ classification)



        dt.write2dArray(all_phrases_complete, all_phrase_fn)


        print("Created class-all")
        all_phrases_complete = np.asarray(all_phrases_complete).transpose()
        for a in range(len(all_phrases_complete)):
            for v in range(len(all_phrases_complete[a])):
                if all_phrases_complete[a][v] > 1:
                    all_phrases_complete[a][v] = 1

        all_phrases_complete = np.asarray(all_phrases_complete).transpose()

        if make_individual:
            for p in range(len(all_phrases_complete)):
                dt.write1dArray(all_phrases_complete[p], output_folder+"binary/phrases/class-" + phrase_list[p] +
                                "-"+str(only_words_in_x_entities) + "-"+str(words_without_x_entities)+"-"+ classification)



        all_phrase_fn = output_folder + "binary/phrases/" + "class-all-" + str(
            only_words_in_x_entities) + "-" + str(words_without_x_entities) + "-" + classification
        dt.write2dArray(all_phrases_complete, all_phrase_fn)

        print("Created class-all binary")
        counter += 1
        #for p in range(len(all_phrases)):
        #    dt.write1dArray(all_phrases[p], output_folder + file_names[p] + ".txt")


def removeClass(folder_name):
    names = dt.getFns(folder_name)
    for name in names:
        if name[:12] == "class-class-":
            contents = dt.import1dArray(folder_name + name)
            dt.write1dArray(contents, folder_name + name[6:])

#removeClass("D:/Dropbox/PhD/My Work/Code/Paper 2/data/movies/bow/ppmi/")


def getAvailableEntities(entity_names_fns, data_type, classification):
    entity_names = []
    for e in entity_names_fns:
        entity_names.append(dt.import1dArray(e))
    dict = {}
    for entity_name in entity_names:
        for name in entity_name:
            dict[name] = 0
    available_entities = []
    for key in dict:
        available_entities.append(key)
    dt.write1dArray(available_entities, "../data/"+data_type+"/classify/"+classification+"available_entities.txt")


from sklearn.feature_extraction.text import TfidfTransformer

def convertPPMI(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i, :] = 0 \
            if rowTotals[0,i] == 0 \
            else rowMat[i, :] * (1.0 / rowTotals[0,i])
        print(i)
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:,j] = 0 if colTotals[0,j] == 0 else (1.0 / colTotals[0,j])
        print(j)
    mat = mat.toarray()
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))
    return P

def convertToTfIDF(data_type, lowest_count, highest_count, freq_arrays_fn, class_type):
    freq = np.asarray(dt.import2dArray(freq_arrays_fn))
    v = TfidfTransformer()
    x = v.fit_transform(freq)
    x = x.toarray()
    dt.write2dArray(x, "../data/"+data_type+"/bow/tfidf/class-all-"+str(lowest_count)+"-"+str(highest_count)+"-"+str(class_type))
    dt.writeClassAll("../data/"+data_type+"/bow/tfidf/class-all-"+str(lowest_count)+"-"+str(highest_count)+"-"+str(class_type),
                     "../data/"+data_type+"/bow/names/"+str(lowest_count)+"-"+str(highest_count)+"-"+str(class_type)+".txt",
                  "../data/"+data_type+"/bow/names/"+str(lowest_count)+"-"+str(highest_count)+"-"+str(class_type)+".txt",
                     "../data/"+data_type+"/bow/tfidf/class-all-"+str(lowest_count)+"-"+str(highest_count)+"-"+str(class_type))



def printIndividualFromAll(data_type, type, lowest_count, max,  classification, all_fn=None, names_array = None):
    fn = "../data/" + data_type + "/bow/"
    if all_fn is None:
        all_fn = fn + type + "/class-all-"+str(lowest_count)+"-"+str(max)+"-"+str(classification)
    if names_array is None:
        names = dt.import1dArray(fn + "names/"+str(lowest_count)+"-"+str(max)+"-"+str(classification)+".txt")
    else:
        names = names_array
    with open(all_fn) as all:
        c = 0
        for la in all:
            convert = dt.convertLine(la)
            dt.write1dArray(convert, fn+ type+"/class-"+str(names[c]+"-"+str(lowest_count)+"-"+str(max)+"-"+str(classification)))
            print(c, len(names), names[c])
            c+=1
    print("wrote individual from all")

def writeClassesFromNames(folder_name, file_names, output_folder):
    names = dt.getFolder(folder_name)
    all_names = defaultdict(int)
    entity_names = dt.import1dArray(file_names)
    translator = str.maketrans({key: None for key in string.punctuation})

    for type in range(len(names)):
        for n in range(len(names[type])):
            names[type][n] = dt.removeEverythingFromString(names[type][n])
            all_names[names[type][n]] += 1
    available_class_names = []
    available_indexes = []
    for n in range(len(entity_names)):
        name = entity_names[n]
        original_name = name
        name = dt.removeEverythingFromString(name)
        if all_names[name] > 0:
            available_class_names.append(original_name)
            available_indexes.append(n)
            print(name, "exists")
        else:
            print(name, "FAIL")
    dt.write1dArray(available_indexes, output_folder + "available_indexes.txt")
    dt.write1dArray(available_class_names, output_folder + "available_entities.txt")
    print("Wrote available indexes and entities")
    class_all = []
    for c in range(len(names)):
        binary_class = []
        for n in range(len(available_class_names)):
            available_class_names[n] = dt.removeEverythingFromString(available_class_names[n])
            if available_class_names[n] in names[c]:
                binary_class.append(1)
            else:
                binary_class.append(0)
        dt.write1dArray(binary_class, output_folder + "class-"+str(c)+"")
        class_all.append(binary_class)
    dt.write2dArray(np.asarray(class_all).transpose(), output_folder + "class-all")
    print("Wrote class-all")

def writeFromMultiClass(multi_class_fn, output_folder, entity_names_fn, data_type, classify_name):
    # Get the entities we have phrases for
    entity_names = dt.import1dArray(entity_names_fn)

    # Import multi classes
    multi_class = dt.import1dArray(multi_class_fn)
    class_names = []
    class_val = []
    highest_class = 0

    for line in multi_class:
        cn, cv = re.split(r'\t+', line)
        cv = int(cv)
        class_names.append(cn)
        class_val.append(cv)
        if cv  > highest_class:
            highest_class = cv



    matched_entity_names = list(set(entity_names).intersection(class_names))
    matched_entity_names.sort()
    dt.write1dArray(matched_entity_names, "../data/" + data_type + "/classify/"+classify_name+"/available_entities.txt")


    indexes_to_delete = []

    for n in range(len(class_names)):
        found = False
        for en in range(len(matched_entity_names)):
            if class_names[n] == matched_entity_names[en]:
                found=True
                break
        if found is False:
            indexes_to_delete.append(n)

    class_val = np.delete(class_val, indexes_to_delete)

    classes = []
    print("Found " + str(highest_class) + " classes")
    for e in range(len(matched_entity_names)):
        class_a = [0] * highest_class
        class_a[class_val[e]-1] = 1
        classes.append(class_a)
    dt.write2dArray(classes, "../data/"+data_type+"/classify/"+classify_name+"/class-all")
    print("Wrote class all")
    classes = np.asarray(classes).transpose()


    for cn in range(len(classes)):
        dt.write1dArray(classes[cn], "../data/"+data_type+"/classify/"+classify_name+"/class-"+str(cn))
        print("Wrote", "class-"+str(cn))

def removeClass(array_fn):
    array = dt.import1dArray(array_fn)
    for e in range(len(array)):
        array[e] = array[e][6:]
    dt.write1dArray(array, array_fn)

#removeClass("../data/movies/bow/names/top5kof17k.txt")

def trimRankings(rankings_fn, available_indexes_fn, names, folder_name):
    available_indexes = dt.import1dArray(available_indexes_fn)
    rankings = np.asarray(dt.import2dArray(rankings_fn))
    names = dt.import1dArray(names)
    trimmed_rankings = []
    for r in range(len(rankings)):
        trimmed = rankings[r].take(available_indexes)
        trimmed_rankings.append(trimmed)
    for a in range(len(trimmed_rankings)):
        print("Writing", names[a])
        dt.write1dArray(trimmed_rankings[a], folder_name + "class-" + names[a])
    print("Writing", rankings_fn[-6:])
    dt.write2dArray(trimmed_rankings, folder_name + "class-" + rankings_fn[-6:])

def match_entities(entity_fn, t_entity_fn, entities_fn, classification):
    names = dt.import1dArray(entity_fn)
    t_names = dt.import1dArray(t_entity_fn)
    entities = dt.import2dArray(entities_fn)
    indexes_to_delete = []
    amount_found = 0
    for n in range(len(names)):
        names[n] = dt.removeEverythingFromString(names[n])
    for n in range(len(t_names)):
        t_names[n] = dt.removeEverythingFromString(t_names[n])
    matched_ids = []
    for n in range(len(t_names)):
        for ni in range(len(names)):
            matched_name = t_names[n]
            all_name = names[ni]
            if matched_name == all_name:
                print(matched_name)
                matched_ids.append(ni)
                break
    matched_entities = []
    for e in matched_ids:
        matched_entities.append(entities[e])

    print("Amount found", amount_found)
    dt.write2dArray(matched_entities, entities_fn[:len(entities_fn)-4] + "-" + classification + ".txt")

# Parsing tree in this format

"""
	labyrinth
	DELETE
		cairn
	border
		boundary line
			state line
		DELETE
			shoreline
				beach
					sandy beach
					topless beach
						nude beach
				coastline
					foreshore
			wetland
				marsh
					salt marsh
"""

# Where everything with an higher indentation than a prior class is a member of that class
# And a class ends once its indentation is met
# So the algorithm will add things to a class until that indentation changes recursively
# For each indented class inside of the main class "site"

def parseTree(tree_fn, output_fn, entity_names_fn):
    data_type = "placetypes"
    class_name = "opencyc"
    entity_names = dt.import1dArray(entity_names_fn)
    with open(tree_fn, "r") as infile:
        tree = [line for line in infile]
    tree = tree[1:]
    indexes_to_delete = []
    for l in range(len(tree)):
        tree[l] = re.sub(r'\s\*', ' ', tree[l])
        if "DELETE" in tree[l]:
            indexes_to_delete.append(l)

    tree = np.delete(tree, indexes_to_delete)
    entities_classes = {}

    for l in range(len(tree)):
        removed_asterisk = re.sub(r'\*', ' ', tree[l])
        stripped = removed_asterisk.strip()
        entities_classes[stripped] = []

    classes = []
    current_tabs = 0
    current_tabs_index = 0
    current_tab_class = []

    class_names = []
    next_index = 0
    for l in range(len(tree)-1):
        removed_asterisk = re.sub(r'\*', ' ', tree[l])
        entity = removed_asterisk.strip()

        tabs = len(tree[l]) - len(tree[l].strip())
        next_tabs = len(tree[l+1]) - len(tree[l+1].strip())
        print("TRY", entity, tabs, next_tabs)
        # If the tree has a subclass
        if (next_tabs) > tabs and tabs <= 4:
            print("START", entity, tabs, next_tabs)
            for j in range(l+1, len(tree)):
                inner_tabs = len(tree[j]) - len(tree[j].strip())
                removed_asterisk = re.sub(r'\*', ' ', tree[j])
                inner_entity = removed_asterisk.strip()
                print("ADD", inner_entity)
                if inner_tabs <= tabs:
                    print("END", inner_tabs, tabs)
                    break
                else:
                    entities_classes[entity].append(inner_entity)
                    print("found", inner_entity, "added to", entity)

    found_entities = []
    found_arrays = []
    class_names = []
    for key, value in list(entities_classes.items()):
        if len(value) < 30:
            del entities_classes[key]
            continue
        """ Removing entities that aren't in a list
        found = False
        for e in entity_names:
            if key == e:
                found = True
        if not found:
            del entities_classes[key]
            continue
        """
        for v in value:
            found_entities.append(v)
        found_arrays.append(value)
        class_names.append(key)
    found_entities = np.unique(np.asarray(found_entities))
    dt.write1dArray(found_entities, "../data/"+data_type+"/classify/"+class_name+"/available_entities.txt")

    # Sort keys and values
    index = np.argsort(class_names)
    sorted_class_names = []
    sorted_value_names = []
    for i in index:
        sorted_class_names.append(class_names[i])
        sorted_value_names.append(found_arrays[i])
    value_indexes = []
    # Convert values to indexes
    for v in range(len(sorted_value_names)):
        value_index = []
        for g in range(len(sorted_value_names[v])):
            for e in range(len(found_entities)):
                if sorted_value_names[v][g] == found_entities[e]:
                    value_index.append(e)
        value_indexes.append(value_index)

    matrix = np.asarray([[0]* len(entities_classes)]*len(found_entities))
    for c in range(len(sorted_class_names)):
        print(c)
        print("-------------------")
        for v in value_indexes[c]:
            print(v)
            matrix[v, c] = 1
        dt.write1dArray(matrix[c], "../data/placetypes/classify/opencyc/class-" + sorted_class_names[c])

    matrix = np.asarray(matrix)

    dt.write2dArray(matrix, "../data/placetypes/classify/opencyc/class-all")


import pickle
def importCertificates(cert_fn, entity_name_fn):
    all_lines = dt.import1dArray(cert_fn)[14:]
    en = dt.import1dArray(entity_name_fn)
    original_en = dt.import1dArray(entity_name_fn)
    en_name = []
    en_year = []
    for e in range(len(en)):
        split = en[e].split()
        en_year.append(split[len(split)-1])
        name = "".join(split[:len(split)-1])
        en_name.append(dt.removeEverythingFromString(name))


    # Initialize ratings dict
    ratings = {
        "USA:G": [],
        "USA:PG": [],
        "USA:PG-13": [],
        "USA:R": []
    }
    """
    "UK:PG": [],
    "UK:12": [],
    "UK:12A": [],
    "UK:15": [],
    "UK:18": [],
    """
    all_ratings = defaultdict(list)
    recently_found_name = ""
    recently_found_year = ""
    recently_found_found = False
    counter = 0

    temp_fn = "../data/temp/us_cert_dict.pickle"

    if dt.fileExists(temp_fn) is False:
        for line in all_lines:
            line = line.split("\t")
            split_ny = line[0].split("{")[0]
            split_ny = split_ny.split()
            for i in range(len(split_ny)-1, -1, -1):
                if "{" in split_ny[i]:
                    del split_ny[i]
            entity_year_bracketed = split_ny[len(split_ny)-1]

            if "(V)" in entity_year_bracketed or "(TV)" in entity_year_bracketed or "(VG)" in entity_year_bracketed:
                entity_year_bracketed = split_ny[len(split_ny) - 2]
            try:
                entity_year = dt.keepNumbers(entity_year_bracketed)[0]
                entity_name = dt.removeEverythingFromString("".join(split_ny[:len(split_ny)-1]))
                found = False
                skip = False
                if recently_found_name == entity_name and recently_found_year == entity_year:
                    skip = True
                    found = recently_found_found
                if not skip:
                    if not found:
                        for n in range(len(en_name)):
                            if entity_name == en_name[n] and entity_year == en_year[n]:
                                print("found", entity_name, entity_year)
                                found = True
                                break
                if found:
                    if("(" not in line[len(line)-1]):
                        entity_rating = line[len(line)-1]
                    else:
                        entity_rating = line[len(line)-2]
                    all_ratings[entity_rating].append(entity_name)
                    if entity_rating in ratings:
                        ratings[entity_rating].append(entity_name)
                        print("rating correct", entity_name, entity_year, entity_rating)
            except IndexError:
                print("IndexError")
                print(line)
                print(split_ny)
                print(entity_year_bracketed)
            recently_found_name = entity_name
            recently_found_year = entity_year
            recently_found_found = found
            counter += 1
            if counter % 1000 == 0:
                    print(counter)
        # Store data (serialize)
        with open(temp_fn, 'wb') as handle:
            pickle.dump(ratings, handle, protocol=pickle.HIGHEST_PROTOCOL)        # Store data (serialize)
        with open("../data/temp/us_cert_dict_all.pickle", 'wb') as handle:
            pickle.dump(all_ratings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load data (deserialize)
    with open(temp_fn, 'rb') as handle:
        ratings = pickle.load(handle)
    if dt.fileExists("../data/temp/us_cert_dict_all.pickle"):
        with open("../data/temp/us_cert_dict_all.pickle", 'rb') as handle:
            all_ratings = pickle.load(handle)

    top_size = 0
    for key, value in all_ratings.items():
        top_size += len(value)
    print(top_size)
    top_size = 0

    new_ratings = defaultdict(list)
    real_name_dict_fn = "../data/temp/us_real_name_dict.dict"
    if dt.fileExists(real_name_dict_fn) is False:
        # Match the names back to the original names
        for key, value in all_ratings.items():
            for r in ratings:
                if r == key:
                    top_size += len(value)
                    for v in range(len(value)):
                        found = False
                        for n in range(len(en_name)):
                            if value[v] == en_name[n]:
                                found = True
                                value[v] = original_en[n]
                                break
                        if found:
                            new_ratings[key].append(value[v])
                    break
        with open(real_name_dict_fn, 'wb') as handle:
            pickle.dump(new_ratings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(real_name_dict_fn, 'rb') as handle:
            new_ratings = pickle.load(handle)
                # Get the final dict setup
    final_dict = {
        "USA-G": [],
        "USA-PG-PG13": [],
        "USA-R": [],
    }
    """
    "UK-PG": [],
    "UK-12-12A": [],
    "UK-18": [],
    """

    # Append the final dict ratings
    """
    final_dict["UK-PG"].extend(all_ratings["UK:PG"])
    final_dict["UK-12-12A"].extend(all_ratings["UK:12"])
    final_dict["UK-12-12A"].extend(all_ratings["UK:12A"])
    final_dict["UK-12-12A"].extend(all_ratings["UK:15"])
    final_dict["UK-18"].extend(all_ratings["UK:18"])
    """
    final_dict["USA-G"].extend(all_ratings["USA:G"])
    final_dict["USA-PG-PG13"].extend(all_ratings["USA:PG"])
    final_dict["USA-PG-PG13"].extend(all_ratings["USA:PG13"])
    final_dict["USA-R"].extend(all_ratings["USA:R"])

    final_name_dict = {
        "USA-G": [],
        "USA-PG-PG13": [],
        "USA-R": [],

    }

    """
    "UK-PG": [],
    "UK-12-12A": [],
    "UK-18": [],
    """

    # Append the final dict good names
    """
    final_name_dict["UK-PG"].extend(new_ratings["UK:PG"])
    final_name_dict["UK-12-12A"].extend(new_ratings["UK:12"])
    final_name_dict["UK-12-12A"].extend(new_ratings["UK:12A"])
    final_name_dict["UK-12-12A"].extend(new_ratings["UK:15"])
    final_name_dict["UK-18"].extend(new_ratings["UK:18"])
    """
    final_name_dict["USA-G"].extend(new_ratings["USA:G"])
    final_name_dict["USA-PG-PG13"].extend(new_ratings["USA:PG"])
    final_name_dict["USA-PG-PG13"].extend(new_ratings["USA:PG13"])
    final_name_dict["USA-R"].extend(new_ratings["USA:R"])


    # Create a unique list of the entities found
    entities_found = []
    for key, items in new_ratings.items():
        for i in items:
            entities_found.append(i)
    entities_found = np.unique(entities_found)
    print(len(entities_found))


    # Get the en_names back...
    jacked_up_entities_found = []
    for n in entities_found:
        new_n = n.split()[:-1]
        jacked_up_entities_found.append(dt.removeEverythingFromString(" ".join(new_n)))

    classes = [[0]*len(entities_found),[0]*len(entities_found),[0]*len(entities_found)]
    counter = 0
    class_names = []
    for key, items in final_dict.items():
        for i in items:
            for e in range(len(jacked_up_entities_found)):
                if i == jacked_up_entities_found[e]:
                    classes[counter][e] = 1
        class_names.append(key)
        dt.write1dArray(classes[counter], "../data/movies/classify/us-ratings/class-" +key)
        counter += 1

    classes = np.asarray(classes).transpose()

    indexes_to_delete = []

    for c in range(len(classes)):
        found = False
        for i in classes[c]:
            if i == 1:
                found = True
                break
        if not found:
            indexes_to_delete.append(c)

    classes = np.delete(classes, indexes_to_delete, axis=0)
    entities_found = np.delete(entities_found, indexes_to_delete)

    dt.write2dArray(classes, "../data/movies/classify/us-ratings/class-all")
    dt.write1dArray(entities_found, "../data/movies/classify/us-ratings/available_entities.txt")
    dt.write1dArray(class_names, "../data/movies/classify/us-ratings/names.txt")
    print("k")

    #Merge 12/12A

def getTop250Movies(entity_names):
    imdb = dt.import1dArray("../data/raw/imdb/ratings/ratings.list")[28:278]
    orig_en = entity_names
    for e in range(len(entity_names)):
        entity_names[e] = "".join(entity_names[e].split()[:-1])
        entity_names[e] = dt.removeEverythingFromString(entity_names[e])
    top_en = []

    for string in imdb:
        string =string.split(".")[1][1:]
        string =string.split()[:-1]
        string = " ".join(string)
        string = dt.removeEverythingFromString(string)
        top_en.append(string)
    matched_index = []
    for e in range(len(entity_names)):
        for x in range(len(top_en)):
            if entity_names[e] == top_en[x]:
                matched_index.append(e)
                print(entity_names[e])
                break
    dt.write1dArray(matched_index, "../data/movies/top_imdb_indexes.txt")
"""
entity_name_fn = "../data/movies/nnet/spaces/entitynames.txt"
entity_names = dt.import1dArray(entity_name_fn)
getTop250Movies(entity_names)
"""
def convertEntityNamesToIDS(ID_fn, all_names_fn, individual_names_fn, output_fn):
    ID_fn = dt.import1dArray(ID_fn)
    all_names_fn = dt.import1dArray(all_names_fn)
    individual_names_fn = dt.import1dArray(individual_names_fn)
    indexes = []

    for n in range(len(all_names_fn)):
        for name in individual_names_fn:
            if all_names_fn[n] == name:
                indexes.append(n)
    dt.write1dArray(np.asarray(ID_fn)[indexes], output_fn)


cert_fn = "../data/raw/imdb/certs/certificates.list"
entity_name_fn = "../data/movies/nnet/spaces/entitynames.txt"
#importCertificates(cert_fn, entity_name_fn)
"""
convertEntityNamesToIDS("../data/raw/previous work/filmIds.txt", entity_name_fn, "../data/movies/classify/ratings/available_entities.txt",
                        "../data/movies/classify/ratings/entity_ids.txt")

"""
"""
parseTree("../data/raw/previous work/placeclasses/CYCClasses.txt", "../data/placetypes/classify/OpenCYC/",
          "../data/placetypes/classify/OpenCYC/names.txt")
"""
"""

fns = "../data/movies/classify/genres/class-all"
remove_indexes([80, 8351, 14985], fns)

fns = "../data/movies/classify/keywords/class-all"
remove_indexes([80, 8351, 14985], fns)
"""
"""
classification = "types"
data_type = "wines"

match_entities("../data/"+data_type+"/nnet/spaces/entitynames.txt",
    "../data/"+data_type+"/classify/"+classification+"/available_entities.txt",
               "../data/"+data_type+"/nnet/spaces/wines100.txt", classification)

classification = "geonames"
data_type = "placetypes"

match_entities("../data/"+data_type+"/nnet/spaces/entitynames.txt",
    "../data/"+data_type+"/classify/"+classification+"/available_entities.txt",
               "../data/"+data_type+"/rank/numeric/places100projected.txt", classification)
"""
"""
writeFromMultiClass("../data/raw/previous work/placeclasses/GeonamesClasses.txt", "../data/placetypes/classify/Geonames/",
                    "../data/raw/previous work/placeNames.txt", data_type="placetypes", classify_name="Geonames")

writeFromMultiClass("../data/raw/previous work/placeclasses/Foursquareclasses.txt", "../data/placetypes/classify/Foursquare/",
                    "../data/raw/previous work/placeNames.txt", data_type="placetypes", classify_name="Foursquare")
"""
"""
match_entities("../data/"+data_type+"/nnet/spaces/entitynames.txt",
    "../data/"+data_type+"/classify/"+classification+"/available_entities.txt",
               "../data/"+data_type+"/nnet/spaces/films100.txt", classification)
"""
"""
"""
"""
classification = "keywords"
data_type = "movies"

match_entities("../data/"+data_type+"/nnet/spaces/entitynames.txt", "../data/"+data_type+"/classify/"+classification+"/available_entities.txt",
               "../data/"+data_type+"/nnet/spaces/films200.txt", classification)
"""
"""
data_type = "wines"
output_folder = "../data/"+data_type+"/classify/types/"
folder_name = "../data/raw/previous work/wineclasses/"
file_names = "../data/"+data_type+"/nnet/spaces/entitynames.txt"
phrase_names = "../data/"+data_type+"/bow/names/50-10-types.txt"
writeClassesFromNames(folder_name, file_names, output_folder)

folder_name = "../data/"+data_type+"/bow/binary/phrases/"
exit()
"""
#trimRankings("../data/movies/nnet/spaces/films200.txt", "../data/"+data_type+"/classify/genres/available_indexes.txt", phrase_names, folder_name)

"""
min=10
max=1
class_type = "movies"
classification = "keywords"
raw_fn = "../data/raw/previous work/movievectors/tokens/"
extension = "film"
cut_first_line = False
get_all = False
additional_name = ""
make_individual = True
"""
def main(min, max, data_type, raw_fn, extension, cut_first_line, additional_name, make_individual, entity_name_fn,
         use_all_files, sparse_matrix, word_count_amt, classification):

    getVectors(raw_fn, entity_name_fn, extension, "../data/"+data_type+"/bow/",
           min, max, cut_first_line, get_all, additional_name,  make_individual, classification, use_all_files, 1000, data_type,
               sparse_matrix)

    bow = sp.csr_matrix(dt.import2dArray("../data/"+data_type+"/bow/frequency/phrases/class-all-"+str(min)+"-" + str(max)+"-"+classification))
    dt.write2dArray(convertPPMI( bow), "../data/"+data_type+"/bow/ppmi/class-all-"+str(min)+"-"+str(max)+"-" + classification)

    print("indiviual from all")
    printIndividualFromAll(data_type, "ppmi", min, max,  classification)

    printIndividualFromAll(data_type, "binary/phrases", min, max,  classification)

    convertToTfIDF(data_type, min, max, "../data/"+data_type+"/bow/frequency/phrases/class-all-"+str(min)+"-"+str(max)+"-"+classification, classification)

    printIndividualFromAll(data_type, "tfidf", min, max,  classification)


min=50
max=10
"""
class_type = "movies"
classification = "all"
raw_fn = "../data/raw/previous work/movievectors/tokens/"
extension = "film"
cut_first_line = True
entity_name_fn = "../data/raw/previous work/filmIds.txt"
use_all_files = None#""
word_count_amt = 0
"""

class_type = "wines"
classification = "types"
raw_fn = "../data/raw/previous work/winevectors/"
extension = ""
cut_first_line = True
use_all_files =  "../data/raw/previous work/winevectors/"
entity_name_fn = "../data/"+class_type+"/nnet/spaces/entitynames.txt"
word_count_amt = 1000
"""
class_type = "placetypes"
classification = "all"
raw_fn = "../data/raw/previous work/placevectors/"
extension = "photos"
cut_first_line = False
entity_name_fn = "../data/"+class_type+"/nnet/spaces/entitynames.txt"
use_all_files = None#""
word_count_amt = 0
"""
get_all = False
additional_name = ""
#make_individual = True
make_individual = True
sparse_matrix = False
print("??")

if  __name__ =='__main__':main(min, max, class_type, raw_fn, extension, cut_first_line, additional_name, make_individual, entity_name_fn, use_all_files,
                               sparse_matrix, word_count_amt, classification)


"""
dt.write2dArray(convertPPMI( sp.csr_matrix(dt.import2dArray("../data/wines/bow/frequency/phrases/class-all-50"))), "../data/wines/bow/ppmi/class-all-50")
dt.write2dArray(convertPPMI( sp.csr_matrix(dt.import2dArray("../data/movies/bow/frequency/phrases/class-all-100"))), "../data/movies/bow/ppmi/class-all-100")
"""
#convertToTfIDF("wines", 50, "../data/wines/bow/frequency/phrases/class-all-50")
#convertToTfIDF("movies", 100, "../data/movies/bow/frequency/phrases/class-all-100")

"""
printIndividualFromAll("placetypes", "tfidf", lowest_count)
printIndividualFromAll("wines", "ppmi", lowest_count)
printIndividualFromAll("wines", "tfidf", lowest_count)
printIndividualFromAll("movies", "ppmi", lowest_count)
printIndividualFromAll("movies", "tfidf", lowest_count)
"""