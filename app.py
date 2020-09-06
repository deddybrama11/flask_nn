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

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron["weights"], inputs)
            neuron["output"] = transfer(activation)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column),max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])

def predict(network, row):
    outputs = forward_propagate(network, row)
    # print(outputs)
    # print(outputs.index(max(outputs)))
    return outputs.index(max(outputs))

connection = mysql.connector.connect(host='sql12.freemysqlhosting.net',
                                         database='sql12363598',
                                         user='sql12363598',
                                         password='jdBnb963Jv')

db_Info = connection.get_server_info()
print("Connected ke Mysql Server versi ",db_Info)
cursor = connection.cursor()
cursor.execute("select database();")
record = cursor.fetchone()
print("You're connected to database: ",record)



curs = connection.cursor()
curs.execute("SELECT * FROM backprop LIMIT 1")
records = cursor.fetchone()
tmp = dict
for x in records:
    tmp = x
    # print(tmp)
tmp = json.loads(tmp)
print(tmp['neural'])


curs.execute("SELECT * FROM `dataset`")



dataset = [[i for i in item] for item in curs.fetchall()]
print(dataset)
curs.close()
#API
app = Flask(__name__)

#route http post to this method
@app.route('/api/upload',methods=['POST'])
def upload():
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    dataset_sementara = copy.deepcopy(dataset)
    #process image
    img_processed = imgProcessing(img)

    dataset_sementara.append(img_processed)
    # print('dataset_coba2 : ', dataset_sementara)


    mm= dataset_minmax(dataset_sementara)
    # print(mm)
    normalize_dataset(dataset_sementara, mm)

    #=========================================================================================================================#

    # print('normal dataset : ', dataset)
    # print('dataset_coba2 after minmax: ', dataset_sementara)
    img_p = dataset_sementara[-1]
    # print('img processed after : ', img_p)


    # mm = minmax_one(img_processed)
    img_p[-1]=None
    # print(img_p)
    # print()
    # normalize(img_processed,mm)
    # print(img_processed)


    p = str(predict(tmp['neural'], img_p))
    # print(p)
    # path_file = ('static/%s.jpg' %uuid.uuid4().hex)
    # cv2.imwrite(path_file, img)
    # kirimReq = [path_file, p]
    return Response(response=p,status=200,mimetype="application/json") # return json string

@app.route('/api/diagnosa/<int:kode>',methods=['GET'])
def diagnosa(kode):
    print(kode)
    mycursor = connection.cursor()
    sql = "SELECT master_diagnosa.nama_diagnosa, daftar_diagnosa.deskripsi_diagnosa, daftar_diagnosa.pencegahan FROM daftar_diagnosa,master_diagnosa WHERE daftar_diagnosa.kode_diagnosa=master_diagnosa.kode_diagnosa AND master_diagnosa.kode_diagnosa='%s'" %kode;
    mycursor.execute(sql)
    recordz = mycursor.fetchall()
    mycursor.close()
    response = app.response_class(
        response=json.dumps(recordz),
        status = 200,
        mimetype='application/json'
    )
    return {"penyakit":recordz[0][0],"deskripsi":recordz[0][1], "pencegahan":recordz[0][2]},200


@app.route('/api/image/<int:kode>',methods=['GET'])
def image(kode):
    mycursor = connection.cursor()
    sql = "SELECT gambar FROM daftar_gambar WHERE kode_diagnosa = '%s'" % kode;
    mycursor.execute(sql)
    recordz = mycursor.fetchall()
    mycursor.close()
    vzs = list()
    vzs.append({"gambar":recordz[0][0]})
    vzs.append({"gambar": recordz[1][0]})
    vzs.append({"gambar": recordz[2][0]})
    response = app.response_class(
        response=json.dumps(vzs),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/api/list_penyakit',methods=['GET'])
def list_penyakit():
    mycursor = connection.cursor()
    sql = "SELECT * FROM `master_diagnosa`";
    mycursor.execute(sql)
    recordz = mycursor.fetchall()
    print(recordz)
    vzs = list()
    mycursor.close()
    for i in recordz:
        vzs.append({"kode_penyakit": i[0],"nama_penyakit":str(i[1])})
    # vzs.append({"kode_penyakit": recordz[0][0],"nama_penykait":recordz[0][1]})
    # vzs.append({"kode_penyakit": recordz[1][0],"nama_penykait":recordz[1][1]})
    # vzs.append({"kode_penyakit": recordz[2][0],"nama_penykait":recordz[2][1]})
    response = app.response_class(
        response=json.dumps(vzs),
        status=200,
        mimetype='application/json'
    )
    return response

#start server
if __name__ == "__main__":
    app.run(debug = True)
