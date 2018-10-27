import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Устанавливаем seed для повторяемости результатов (We can set "seed" for repeating the same results)
#numpy.random.seed(42)

# Размер изображения (Size of a picture)
img_rows, img_cols = 28, 28

# Загружаем данные (Load data from mnist)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений (Image Size Conversion)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# Нормализация данных(Data normalization)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории(Convert labels into categories)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Создаем последовательную модель(Create a sequential model)
model = Sequential()

model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Компилируем модель(Compile the model)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть(Train the network)
model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных(Evaluate the quality of network training on test data)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

#Генерируем описание модели в формате json(Generate the description of the model in json format)
model_json = model.to_json()

#Записываем модель в файл(Write the model to the file)
json_file = open("mnist_cnn.json", "w")
json_file.write(model_json)
json_file.close()

#Записывание весов(Write weights to the file)
model.save_weights("mnist_cnn.h5")
