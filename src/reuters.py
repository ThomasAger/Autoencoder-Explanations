# For the ModApte config
def getSplits(vectors, classes, dev=0.8):
    if len(vectors) != 10655 or len(classes) != 10655:
        print(len(vectors), len(classes), "This is not the standard size of reuters, expected 10655 (Duplicates removed)")
        print()
        return False

    x_train = vectors[:7656]
    x_test = vectors[7656:]
    y_train = classes[:7656]
    y_test = classes[7656:]

    print("Returning development splits", dev)

    x_dev = x_train[int(len(x_train) * 0.8):]
    y_dev = y_train[int(len(y_train) * 0.8):]
    x_train = x_train[:int(len(x_train) * 0.8)]
    y_train = y_train[:int(len(y_train) * 0.8)]

    print(len(x_test), len(x_test[0]), "x_test")
    print(len(y_test),  "y_test")
    print(len(x_train), len(x_train[0]), "x_train")
    print(len(y_train),  "y_train")

    return x_train, y_train, x_test, y_test, x_dev, y_dev