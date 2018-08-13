if __name__ == '__main__':
    ### GENERAL

    # If using the development set or the test set
    dev_set = [True]
    cross_val = 5

    ### DIRECTIONS

    # The minimum amount of documents the word must occur
    min_freq = [200]
    # The frequency minus this number is the max amount of documents the word can occur
    max_freq = [10]

    ### SCORING

    # The method of scoring the directions used in the later clustering algorithm
    score_type = ["kappa", "ndcg", "acc"]

    ### NNET PARAMS

    # The amount of epochs the finetuning network is ran for
    epochs = [300]

    ### CLUSTER PARAMS

    # The clustering algorithm to use, kmeans/meanshift are the scikit-learn implementations
    cluster_type = ["kmeans"]  # "derrac", "meanshift"
    # The share that word-vectors have when averaged with the directions, e.g. 1=all word vectors, 0=no word vectors
    word_vectors = [0.5]

    ## derrac
    # The amount of clusters
    cluster_centers = [200]
    # The amount of directions used to form the cluster centers
    cluster_center_directions = [400]
    # The amount of directions clustered with the centers
    cluster_directions = [2000]

    ## meanshift
    # The parameter for the distance between points, uses estimate bandwidth and then modifies it
    bandwidth = []  # amount estimate bandwidth is multiplied by

    ## kmeans
    # Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    n_init = [10]
    # Maximum number of iterations of the k-means algorithm for a single run.
    max_iter = [300]



    pipeline()
