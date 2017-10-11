
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize
import os
from keras.models import model_from_json
json_file = open('cifar10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("cifar.h5")
def predict_test(test_file, model):
    class_info =  ['bird','airplane', 'automobile', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # class_info = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    image_data = imread(test_file)
    image_resized = imresize(image_data, (32, 32))
    image_arr = np.asarray(image_resized.transpose(2, 0, 1), dtype='float32')
    dataset = np.ndarray((1, 3, 32, 32), dtype=np.float32)
    dataset[0, :, :, :] = image_arr
    dataset /= 255
    pred_pr = model.predict_proba(dataset, verbose=0)
    pred_cl = model.predict_classes(dataset, verbose=0)[0]


    # print('Predicted Class:', class_info[pred_cl])
    # print( 'Predicted Probabilities:',pred_pr[0][i])

    for i in range(0, 10):
        print(class_info[i], ':', pred_pr[0][i])
    return class_info[pred_cl]
print(predict_test('dog.jpg',model))
