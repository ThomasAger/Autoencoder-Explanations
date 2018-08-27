def averageWordVectors(id2word, ppmi_fn, size, data_type):
    bow = import2dArray(ppmi_fn)

    if len(bow[0]) != len(id2word.keys()):
        print("vocab and bow dont match", len(bow[0]), len(id2word.keys()))
        exit()
    print("Creating dict")
    print("Importing word vectors")
    glove_file = "../data/raw/glove/glove.6B." + str(size) + 'd.txt'
    tmp_file = "../data/raw/glove/test_word2vec.txt"
    glove2word2vec(glove_file, tmp_file)

    all_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    print("Creating vectors")
    vectors = []
    i = 0
    for doc in bow:
        to_average = []
        for w in range(len(doc)):
            if doc[w] > 0:
                try:
                    to_average.append(np.multiply(all_vectors.get_vector(id2word[w]), doc[w]))
                except KeyError:
                    print("keyerror", id2word[w])
        ppmi_sum = np.sum(doc)
        print("ppmi sum", ppmi_sum)
        if len(to_average) == 0:
            vectors.append(np.zeros(shape=size))
            print("FAILED", i, "words:", len(to_average))
            continue
        else:
            print(i, "words:", len(to_average), "dim", len(to_average[0]))
        sum = np.sum(to_average, axis=0)
        vector = np.divide(sum, ppmi_sum)
        vectors.append(vector)
        for i in range(len(vector)):
            if math.isnan(vector[i]):
                print("ok", i,  sum, vector, vector[i])
                break
        i+=1

    np.save("../data/" +data_type+"/nnet/spaces/wvPPMIFIXED" + str(size) + ".npy", vectors)