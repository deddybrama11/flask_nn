from math import exp
from random import seed
from random import random
from csv import reader
from testa import imgProcessing
from random import randrange
import cv2
import time
import uuid
import jsonify
import copy
import json
import numpy as np
from flask import Flask, request, Response
from json import dumps
import mysql.connector
from mysql.connector import Error

connection = mysql.connector.connect(host='localhost',
                                         database='neural_network_tomat',
                                         user='root',
                                         password='')

db_Info = connection.get_server_info()
print("Connected ke Mysql Server versi ",db_Info)
cursor = connection.cursor()
cursor.execute("select database();")
record = cursor.fetchone()
print("You're connected to database: ",record)




start_time = time.time()

array_of_network = []
dataset_coba = list()
dataset_coba2 = list()

#initialize network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{"weights":[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{"weights":[random() for i in range(n_hidden +1)]} for i in range(n_outputs)]
    network.append(output_layer)
    print(network)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron["weights"], inputs)
            neuron["output"] = transfer(activation)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    print(row)
    print(network)
    print()
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # print("len network = ",len(network))
        # print("i = ",i)
        if i != len(network)-1:
            # print("11")
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    # print(neuron[how weights'][j])
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
        else:
            # print("22")
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])
    # print("-------")

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i !=0:
            inputs = [neuron["output"] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron["weights"][j]+= l_rate * neuron["delta"] * inputs[j]
            neuron["weights"][-1] += l_rate * neuron["delta"] #update untuk BIAS


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
    outputs = forward_propagate(network, row)
    # print(outputs)
    # print(outputs.index(max(outputs)))
    return outputs.index(max(outputs))

def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):

    look = {'Sehat':0,'Yellow Leaf Curl':1, 'Septoria':2}
    for row in dataset:
        row[column] = look[row[column]]


    #before edit
    # class_values = [row[column] for row in dataset]
    # # print(class_values)
    # unique = set(class_values)
    # # print(unique)
    # lookup = dict()
    # for i, value in enumerate(unique):
    #     # print(value,' = ',i)
    #     lookup[value] = i
    # print('Lookup1',look)
    # print('Lookup ',lookup)
    # for row in dataset:
    #     row[column] = lookup[row[column]]
    # # for row in dataset:
    #     # print(row)

def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column),max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
        # print(row)

def minmax_one(fl):
    minmax = list()
    stats = [min(fl),max(fl)]
    print(stats)
    return stats

def normalize(fl, minmax):
    for i in range(len(fl)-1):
        fl[i] = (fl[i]-minmax[0])/(minmax[1]-minmax[0])
    print(fl)

#split datset ke k folds
def cross_validation_split(dataset, n_folds):
    class_values = [row[len(dataset[0])-1] for row in dataset]
    # print(class_values)
    unique = set(class_values)
    # print(unique)
    lookup = dict()
    for i, value in enumerate(unique):
        # print(value,' = ',i)
        lookup[value] = i
    print('Lookup ', lookup)
    print('panjang lookup ',len(lookup))

    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)*n_folds)
    print(len(dataset)*n_folds)

    fold = list()
    while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
    print()
    train = list()
    data_train = int(len(dataset) - fold_size)
    while len(train) < data_train:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    dataset_split.append(train)

    print('After : ',dataset_split)

    # dataset_split2 = list()
    # dataset_copy2 = list(dataset)
    # dtest = list()
    # tst1 = 88
    # tst2 = 176
    # tst3 = 264
    #
    # for i in range(0,17):
    #     dtest.append(dataset_copy2.pop(i))
    # for i in range(tst1,(tst1+17)):
    #     dtest.append(dataset_copy2.pop(i))
    # for i in range(tst2, (tst2+17)):
    #     dtest.append(dataset_copy2.pop(i))
    # dataset_split2.append(dtest)
    # print("===+=====")
    # print(dataset_split2)
    # for i in dataset_split2[0]:
    #     print(i)
    # dataset_split2.append(dataset_copy2)
    # print("panjang dataset copy 2 == ",len(dataset_copy2))


    # train2 = list()
    # data_train2 = int(len(dataset) - len(dtest))
    # while len(train2) < data_train2:
    #     index = randrange(len(dataset_copy))
    #     train.append(dataset_copy.pop(index))
    # dataset_split.append(train)




    # for ds  in dataset_split:
    #     print(len(ds))
    return dataset_split

#kalkulasi persen akurasi
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct+=1
    return correct / float(len(actual))*100.0

#evaluasi algoritma menggunakan cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    print(folds)
    dataset = folds
    dataset_coba.append(folds.copy())
    # print(folds)
    scores = list()
    for fold in folds:
        print("Length fold : ",len(fold))
        train_set = list(folds)
        train_set.remove(fold)
        print("jumlah trainset setelah remove = ",len(train_set[0]))
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        # print(predicted)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        break
    return scores

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    #buat sendiri array of network
    array_of_network.append(network)
    predictions=list()
    for row in test:
        prediction = predict(network,row)
        predictions.append(prediction)
    return(predictions)


#test backprop on seeds dataset
# seed(1)
#load and prepare data
filename = "coba.csv"
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    print(i)
    str_column_to_float(dataset,i)
str_column_to_int(dataset, len(dataset[0])-1)
dataset_coba2 = copy.deepcopy(dataset)
mycursor = connection.cursor()
for row in dataset_coba2:
    sql = "INSERT INTO dataset VALUES(%f,%f,%f,%f,%f,%f,%d)" % (row[0],row[1],row[2],row[3],row[4],row[5],row[6]);
    # print(sql)
    mycursor.execute(sql)
    connection.commit()
    # print(mycursor.rowcount, "record inserted.")
mycursor.close()

minmax = dataset_minmax(dataset)
print(minmax)
normalize_dataset(dataset, minmax)
n_folds = 0.2
l_rate = 0.1
n_epoch = 4000
n_hidden = 12
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' %scores)
print('Mean accuracyL:%.3f%%' % (sum(scores)/float(len(scores))))
print()
for zs in array_of_network:
    print(zs)
print()
for pl in array_of_network[0]:
    print(pl)
print('================================')
print("panjang dataset coba[0] : ",len(dataset_coba[0]))
print("panjang dataset coba[0][1] : ",len(dataset_coba[0][0]))
print("panjang dataset coba[0][2] : ",len(dataset_coba[0][1]))
print()
print(dataset_coba2)
print()

print("--- %s seconds ---" % (time.time() - start_time))
print()


# n_folds = 0.2
# l_rate = 0.3
# n_epoch = 1000
# n_hidden = 20


#INSERT ke database MYSQL
nn = {"neural" : array_of_network[0]}
mycursor = connection.cursor()
sql = "INSERT INTO backprop VALUES('%s')"%(json.dumps(nn));
print(sql)
mycursor.execute(sql)
connection.commit()
print(mycursor.rowcount, "record inserted.")
mycursor.close()

print()
#ambil data NN dari DB
curs = connection.cursor()
curs.execute("SELECT * FROM backprop LIMIT 1")
records = cursor.fetchone()
tmp = dict
for x in records:
    tmp = x
    print(tmp)
tmp = json.loads(tmp)
#debugging
print("db : ",tmp['neural'])
print("asl: ",array_of_network[0])
print("db_test : ",tmp['neural'][0][0]['weights'])
print("asl_test: ",array_of_network[0][0][0]['weights'])

curs.close()


# ========================= ===================================================
# #API
# app = Flask(__name__)
#
# #route http post to this method
# @app.route('/api/upload',methods=['POST'])
# def upload():
#     img = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
#     dataset_sementara = copy.deepcopy(dataset_coba2)
#     #process image
#     img_processed = imgProcessing(img)
#
#     dataset_sementara.append(img_processed)
#     print('dataset_coba2 : ', dataset_sementara)
#
#     mm= dataset_minmax(dataset_sementara)
#     print(mm)
#     normalize_dataset(dataset_sementara, mm)
#     #=========================================================================================================================#
#     print('normal dataset : ', dataset)
#     print('dataset_coba2 after minmax: ', dataset_sementara)
#     img_p = dataset_sementara[-1]
#     print('img processed after : ', img_p)
#
#     # mm = minmax_one(img_processed)
#     img_p[-1]=None
#     print(img_p)
#     print()
#     # normalize(img_processed,mm)
#     # print(img_processed)
#
#     p = str(predict(tmp['neural'], img_p))
#     print(p)
#     # path_file = ('static/%s.jpg' %uuid.uuid4().hex)
#     # cv2.imwrite(path_file, img)
#     # kirimReq = [path_file, p]
#     return Response(response=p,status=200,mimetype="application/json") # return json string
#
# @app.route('/api/diagnosa/<int:kode>',methods=['GET'])
# def diagnosa(kode):
#     print(kode)
#     mycursor = connection.cursor()
#     sql = "SELECT master_diagnosa.nama_diagnosa, daftar_diagnosa.deskripsi_diagnosa, daftar_diagnosa.pencegahan FROM daftar_diagnosa,master_diagnosa WHERE daftar_diagnosa.kode_diagnosa=master_diagnosa.kode_diagnosa AND master_diagnosa.kode_diagnosa='%s'" %kode;
#     mycursor.execute(sql)
#     recordz = mycursor.fetchall()
#     response = app.response_class(
#         response=json.dumps(recordz),
#         status = 200,
#         mimetype='application/json'
#     )
#     return {"penyakit":recordz[0][0],"deskripsi":recordz[0][1], "pencegahan":recordz[0][2]},200
#
#
# @app.route('/api/image/<int:kode>',methods=['GET'])
# def image(kode):
#     mycursor = connection.cursor()
#     sql = "SELECT gambar FROM daftar_gambar WHERE kode_diagnosa = '%s'" % kode;
#     mycursor.execute(sql)
#     recordz = mycursor.fetchall()
#     vzs = list()
#     vzs.append({"gambar":recordz[0][0]})
#     vzs.append({"gambar": recordz[1][0]})
#     vzs.append({"gambar": recordz[2][0]})
#     response = app.response_class(
#         response=json.dumps(vzs),
#         status=200,
#         mimetype='application/json'
#     )
#     return response
#
# #start server
# app.run(host="0.0.0.0",port=5000)


# if(connection.is_connected()):
#     cursor.close()
#     connection.close()
#     print("Mysql connection is closed")

