
import data as dt
import re
import numpy as np
import string
from collections import defaultdict
import random
import theano
from theano.tensor.shared_randomstreams import RandomStreams

def getVectors(input_folder, file_names_fn, extension, output_folder, only_words_in_x_entities,
               words_without_x_entities, cut_first_line=False, get_all=False, additional_name="", make_individual=True,
               classification=""):
    file_names = dt.import1dArray(file_names_fn)
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
            for p in phrase_list:
                if p[0] != "all":
                    phrase_dict[p[0]] += 1
                else:
                    print("found class all")
            working_filenames.append(file_names[f])
        except FileNotFoundError:
            print("Failed to find", file_names[f], f)
            failed_filenames.append(file_names[f])
            failed_indexes.append(f)
    print(failed_indexes)
    print(failed_filenames)
    phrase_sets = []
    # Convert to array so we can sort it
    phrase_list = []
    for key, value in phrase_dict.items():
        if value >= only_words_in_x_entities:
            phrase_list.append(key)
    all_phrases = []
    for key, value in phrase_dict.items():
        all_phrases.append(key)

    phrase_sets.append(phrase_list)
    phrase_sets.append(all_phrases)
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

        print("Each entity is length", len(all_phrases_complete[0]))
        print("The overall matrix is", len(all_phrases_complete))

        # Then, populate the overall bag of words for each film (with all other phrases already set to 0

        for f in range(len(working_filenames)):
            n_phrase_list = dt.import2dArray(input_folder + working_filenames[f] + "." + extension, "s")
            if cut_first_line:
                n_phrase_list = n_phrase_list[1:]
            for p in n_phrase_list:
                phrase = p[0]
                try:
                    phrase_index = phrase_index_dict[phrase]
                    all_phrases_complete[f][phrase_index] = int(p[1])
                    #print("Kept", phrase)
                except KeyError:
                    continue
                    #print("Deleted phrase", phrase)

        all_phrases_complete = np.asarray(all_phrases_complete).transpose()

        indexes_to_delete = []
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
                print("Wrote", phrase_list[p])


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
                print("Wrote binary", phrase_list[p])


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
import scipy.sparse as sp
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


def printIndividualFromAll(data_type, type, lowest_count, max, class_type, classification):
    fn = "../data/" + data_type + "/bow/"
    all = np.asarray(dt.import2dArray(fn + type + "/class-all-"+str(lowest_count)+"-"+str(max)+"-"+str(classification)))
    names = dt.import1dArray(fn + "names/"+str(lowest_count)+"-"+str(max)+"-"+str(classification)+".txt")
    for c in range(len(all)):
        dt.write1dArray(all[c], fn+ type+"/class-"+str(names[c]+"-"+str(lowest_count)+"-"+str(max)+"-"+str(classification)))
        print("Wrote " + str(names[c]))

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
    dt.write2dArray(class_all, output_folder + "class-all")
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

def parseTree(tree_fn, output_fn):
    tree = dt.import1dArray(tree_fn)
    tree = tree[1:]
    for l in range(len(tree)):
        if dt.removeEverythingFromString(tree[l]) == "DELETE":
            del tree[l]

    classes = []
    current_tabs = 0
    current_tabs_index = 0
    current_tab_class = []
    for l in range(len(tree)):

        tabs = len(tree[l]) - len(dt.removeEverythingFromString(tree[l]))
        if tabs != current_tabs:
            current_tabs = tabs
            current_tab_class.append(tree[l])

        else:
            current_tabs_index = l




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
classification = "foursquare"
data_type = "placetypes"
"""
writeFromMultiClass("../data/raw/previous work/placeclasses/GeonamesClasses.txt", "../data/placetypes/classify/Geonames/",
                    "../data/raw/previous work/placeNames.txt", data_type="placetypes", classify_name="Geonames")

writeFromMultiClass("../data/raw/previous work/placeclasses/Foursquareclasses.txt", "../data/placetypes/classify/Foursquare/",
                    "../data/raw/previous work/placeNames.txt", data_type="placetypes", classify_name="Foursquare")

match_entities("../data/"+data_type+"/nnet/spaces/entitynames.txt",
    "../data/"+data_type+"/classify/"+classification+"/available_entities.txt",
               "../data/"+data_type+"/rank/numeric/places100projected.txt", classification)
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
phrase_names = "../data/"+data_type+"/bow/names/50.txt"
#writeClassesFromNames(folder_name, file_names, output_folder)

folder_name = "../data/"+data_type+"/bow/binary/phrases/"
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
def main(min, max, class_type, classification, raw_fn, extension, cut_first_line, additional_name, make_individual, entity_name_fn):
    """
    if classification == "all":
        getVectors(raw_fn, entity_name_fn, extension, "../data/"+class_type+"/bow/",
               min, max, cut_first_line, get_all, additional_name, make_individual, classification)
    else:
        getVectors(raw_fn, "../data/"+class_type+"/classify/"+classification+"/available_entities.txt", extension, "../data/"+class_type+"/bow/",
               min, max, cut_first_line, get_all, additional_name, make_individual, classification)
    """
    bow = sp.csr_matrix(dt.import2dArray("../data/"+class_type+"/bow/frequency/phrases/class-all-"+str(min)+"-" + str(max)+"-"+classification))
    dt.write2dArray(convertPPMI( bow), "../data/"+class_type+"/bow/ppmi/class-all-"+str(min)+"-"+str(max)+"-" + classification)

    printIndividualFromAll(class_type, "ppmi", min, max, class_type, classification)
    #printIndividualFromAll(class_type, "binary/phrases", min, max, class_type, classification)

    convertToTfIDF(class_type, min, max, "../data/"+class_type+"/bow/frequency/phrases/class-all-"+str(min)+"-"+str(max)+"-"+classification, classification)

    printIndividualFromAll(class_type, "tfidf", min, max, class_type, classification)


min=50
max=10
"""
class_type = "movies"
classification = "genres"
raw_fn = "../data/raw/previous work/movievectors/tokens/"
extension = "film"
cut_first_line = True
entity_name_fn = "../data/raw/previous work/filmIds.txt"
"""
"""
class_type = "wines"
classification = "all"
raw_fn = "../data/raw/previous work/winevectors/"
extension = ""
cut_first_line = True
"""

class_type = "placetypes"
classification = "foursquare"
raw_fn = "../data/raw/previous work/placevectors/"
extension = "photos"
cut_first_line = False
entity_name_fn = "../data/"+class_type+"/nnet/spaces/entitynames.txt"

get_all = False
additional_name = ""
#make_individual = True
make_individual = True

if  __name__ =='__main__':main(min, max, class_type, classification, raw_fn, extension, cut_first_line, additional_name, make_individual, entity_name_fn)


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