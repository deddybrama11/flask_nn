import random
number = list(range(0,56))
print('number : ',number)



for i in range(len(number)):
    aa = random.choice(number)
    print(number.index(aa))
    number.pop(number.index(aa))
    print(aa)
    print(number)
    print()
print(number)



dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)*n_folds)
    print(len(dataset)*n_folds)

    number = list(range(0,len(dataset)))
    print('number : ',number)
    fold = list()
    while len(fold) < fold_size:
        randnomor = random.choice(number)
        number.pop(number.index(randnomor))
        fold.append(randnomor)
    dataset_split.append(fold)
    print()
    train = list()
    data_train = int(len(dataset) - fold_size)
    while len(train) < data_train:
        randnomor = random.choice(number)
        number.pop(number.index(randnomor))
        train.append(randnomor)
    dataset_split.append(train)
    print('number : ', number)
    print('After : ',dataset_split)