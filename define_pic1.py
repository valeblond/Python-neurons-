import numpy as np
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing import image

#Загружаем сохранённую сеть их файлов (Load saved network from saved files)
json_file = open("mnist_cnn.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mnist_cnn.h5")

#Копилируем модель перед использованием (Compile our model before using)
loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Загружаем картинку из файла (Load a picture from the file)
img_path = '9.png'
img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
x = image.img_to_array(img)

#Преобразуем картинку в массив и нормализуем (Transform our picture into array and normalize it)
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)

#Запускаем распознавание (Run the recognition)
prediction = loaded_model.predict(x)
prediction = np.argmax(prediction, axis=1)
print(prediction)

