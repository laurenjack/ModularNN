import sklearn.neighbors as nb
import data_processor as dp

data_sets = ["20", "30", "60", "00", "05"]

for ds in data_sets:
    # Load the data
    train, test = dp.load_digits("digits/digits" + ds, nn=False)
    # Unzip for knn
    train_in = [x[0].flatten() for x in train]
    train_out = [x[1] for x in train]
    test_in = [x[0].flatten() for x in test]
    test_out = [x[1] for x in test]
    knc = nb.KNeighborsClassifier()
    knc.fit(train_in, train_out)
    print "KNN Classification error for " + ds + ": " + str(1- knc.score(test_in, test_out))
