# Train splits defined by various standards
imdb_train = 25000
newsgroups_train = 11314
reuters_train = 7656
yahoo_train = 25000
movies_train = 9225
placetypes_train = 913

imdb_total = 50000
newsgroups_total = 18846
reuters_total = 10655
yahoo_total = 50000
movies_total = 13978
placetypes_total = 1383


from sklearn.model_selection import KFold

def splitData(features, classes, data_type, dev_percent=0.2):
    if data_type == "imdb" or data_type == "sentiment":
        train_split = imdb_train
        max_size = imdb_total
    elif data_type == "newsgroups":
        train_split = newsgroups_train
        max_size = newsgroups_total
    elif data_type == "reuters":
        train_split = reuters_train
        max_size = reuters_total
    elif data_type == "yahoo" or data_type == "amazon":
        train_split = yahoo_train
        max_size = yahoo_total
    elif data_type == "movies":
        train_split = movies_train
        max_size = movies_total
    elif data_type == "placetypes":
        train_split = placetypes_train
        max_size = placetypes_total
    else:
        print("No data type found")
        return False

    if len(features) != max_size or len(features) != max_size:
        print("This is not the standard size, expected " + str(max_size))
        return False

    x_train = features[:train_split]
    x_test = features[train_split:]
    y_train = classes[:train_split]
    y_test = classes[train_split:]

    x_dev = None
    y_dev = None
    if dev_percent > 0:
        x_dev = x_train[int(len(x_train) * (1 - dev_percent)):]
        y_dev = y_train[int(len(y_train) * (1 - dev_percent)):]
        x_train = x_train[:int(len(x_train) * (1 - dev_percent))]
        y_train = y_train[:int(len(y_train) * (1 - dev_percent))]
        print(len(x_dev), len(x_dev[0]), "x_dev")
        print(len(y_dev),  "y_dev")

    print(len(x_test), len(x_test[0]), "x_test")
    print(len(y_test),  "y_test")
    print(len(x_train), len(x_train[0]), "x_train")
    print(len(y_train),  "y_train")

    return x_train, y_train, x_test, y_test, x_dev, y_dev


def crossValData(cv_splits, features, classes):
    ac_y_train = []
    ac_x_train = []
    ac_x_test = []
    ac_y_test = []
    ac_y_dev = []
    ac_x_dev = []
    kf = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
    for train, test in kf.split(features):
        ac_y_test.append(classes[test])
        ac_y_train.append(classes[train[:int(len(train) * 0.8)]])
        ac_x_train.append(features[train[:int(len(train) * 0.8)]])
        ac_x_test.append(features[test])
        ac_x_dev.append(features[train[int(len(train) * 0.8):len(train)]])
        ac_y_dev.append(classes[train[int(len(train) * 0.8):len(train)]])
        c += 1
    return ac_x_train, ac_y_train, ac_x_test, ac_y_test, ac_x_dev, ac_y_dev