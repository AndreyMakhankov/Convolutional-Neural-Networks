import gdown
import zipfile
import os
from PIL import Image
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    imagePath = 'content/'
    link = 'https://storage.yandexcloud.net/aiueducation/Content/base/l5/middle_fmr.zip'
    dataImages = []
    IMG_WIDTH = 128
    IMG_HEIGHT = 64
    batchSize = 25  # Размер выборки


    def workingWithFiles():
        dataFiles = []  # Cписок путей к файлам картинок
        dataLabels = []  # Список меток классов, соответствующих файлам
        gdown.download(link, None, quiet=True)
        with zipfile.ZipFile('middle_fmr.zip', 'r') as zip_ref:
            zip_ref.extractall(imagePath)
        classList = sorted(os.listdir(imagePath))
        classCount = len(classList)
        fig, axs = plt.subplots(1, classCount, figsize=(25, 5))
        for i in range(classCount):
            car_path = f'{imagePath}{classList[i]}/'
            img_path = car_path + random.choice(os.listdir(car_path))
            axs[i].set_title(classList[i])
            axs[i].imshow(Image.open(img_path))
            axs[i].axis('off')
        plt.show()

        for i in range(classCount):
            className = classList[i]
            classPath = imagePath + className
            classFiles = os.listdir(classPath)
            print(f'Размер класса {className} составляет {len(classFiles)} машин')
            dataFiles += [f'{classPath}/{file_name}' for file_name in classFiles]
            dataLabels += [i] * len(classFiles)
        print('Общий размер базы для обучения:', len(dataLabels))

        for file_name in dataFiles:
            # Открытие и смена размера изображения
            img = Image.open(file_name).resize((IMG_WIDTH, IMG_HEIGHT))
            imgArr = np.array(img)  # Перевод в numpy-массив
            dataImages.append(imgArr)  # Добавление изображения в виде numpy-массива к общему списку

        x_data = np.array(dataImages) / 255  # Перевод общего списка изображений в numpy-массив и нормирование
        y_data = np.array(dataLabels)  # Перевод общего списка меток класса в numpy-массив

        print(f'В массив собрано {len(dataImages)} фотографий следующей формы: {imgArr.shape}')
        print(f'Общий массив данных изображений следующей формы: {x_data.shape}')
        print(f'Общий массив меток классов следующей формы: {y_data.shape}')
        return x_data, y_data, classCount


    x_data, y_data, classCount = workingWithFiles()

    model = Sequential()
    # Первый сверточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))  # (64,128,3) --> (64,128,256) меняется кол-во фильтров
    model.add(BatchNormalization())  # (64,128,256) --> (64,128,256) нет изменений
    # Второй сверточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # (64,128,256) --> (64,128,256) нет изменений
    model.add(MaxPooling2D(pool_size=(
    3, 3)))  # сжимает размер в три раза, с потерей, т.к. padding='valid', глубина не изм. (64,128,256) --> (21,42,256)
    # Третий сверточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # (21,42,256) --> (21,42,256) нет изменений
    model.add(BatchNormalization())  # (21,42,256) --> (21,42,256) нет изменений
    model.add(Dropout(0.2))  # (21,42,256) --> (21,42,256) нет изменений
    # Четвертый сверточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))  # (21,42,256) --> (21,42,256) нет изменений
    model.add(MaxPooling2D(pool_size=(3,
                                      3)))  # сжимает размер в три раза, хотя padding='valid', потерь нет, т.к. делится нацело, глубина не изм. (21,42,256) --> (7,14,256)
    model.add(Dropout(0.2))  # (7,14,256) --> (7,14,256) нет изменений
    # Пятый сверточный блок
    model.add(Conv2D(512, (3, 3), padding='same',
                     activation='relu'))  # (7,14,256) --> (7,14,512) меняется количество фильтров
    model.add(BatchNormalization())  # (7,14,512) --> (7,14,512) нет изменений
    # Шестой сверточный блок
    model.add(
        Conv2D(1024, (3, 3), padding='same', activation='relu'))  # (7,14,1024) --> (7,14,1024) меняется кол-во фильтров
    model.add(MaxPooling2D(pool_size=(
    3, 3)))  # сжимает размер в три раза, с потерей, т.к. padding='valid', глубина не изм. (7,14,1024) --> (2,4,1024)
    model.add(Dropout(0.2))  # (2,4,1024) --> (2,4,1024) нет изменений
    # Блок классификации
    model.add(Flatten())  # слой преобразования многомерных данных в одномерные (2,4,1024) --> (2*4*1024) --> (8192)
    model.add(Dense(2048, activation='relu'))  # полносвязный слой, меняется кол-во нейронов (8192) --> (2048)
    model.add(Dense(4096, activation='relu'))  # полносвязный слой, меняется кол-во нейронов (2048) --> (4096)
    model.add(Dense(classCount,
                    activation='softmax', ))  # выходной полносвязный слой, кол-во нейронов = количество классов (4096) --> (3)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    store_learning = model.fit(x_data,  # ----------------- x_train, примеры набора данных
                               y_data,  # ----------------- y_train, метки примеров набора данных
                               validation_split=0.2,
                               # --- 0.2 - доля данных для валидационной (проверочной) выборки, 1-0.2=0.8 останется в обучающей
                               shuffle=True,
                               # ----------- перемешивание данных для равномерного обучения, соответствие экземпляра и метки сохраняется
                               batch_size=25,
                               # ---------- размер пакета, который обрабатывает нейронка перед одним изменением весов
                               epochs=30,  # -------------- epochs - количество эпох обучения
                               verbose=1)  # -------------- 0 - не визуализировать ход обучения, 1 - визуализировать
    # Создание полотна для рисунка
    plt.figure(1, figsize=(18, 5))

    # Задание первой (левой) области для построения графиков
    plt.subplot(1, 2, 1)
    # Отрисовка графиков 'loss' и 'val_loss' из значений словаря store_learning.history
    plt.plot(store_learning.history['loss'],
             label='Значение ошибки на обучающем наборе')
    plt.plot(store_learning.history['val_loss'],
             label='Значение ошибки на проверочном наборе')
    # Задание подписей осей
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Значение ошибки')
    plt.legend()

    # Задание второй (правой) области для построения графиков
    plt.subplot(1, 2, 2)
    # Отрисовка графиков 'accuracy' и 'val_accuracy' из значений словаря store_learning.history
    plt.plot(store_learning.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(store_learning.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    # Задание подписей осей
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()

    # Фиксация графиков и рисование всей картинки
    plt.show()
    # Генератор изображений
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # Значения цвета меняем на дробные показания
        rotation_range=10,  # Поворачиваем изображения при генерации выборки
        width_shift_range=0.1,  # Двигаем изображения по ширине при генерации выборки
        height_shift_range=0.1,  # Двигаем изображения по высоте при генерации выборки
        zoom_range=0.1,  # Зумируем изображения при генерации выборки
        horizontal_flip=True,  # Включаем отзеркаливание изображений
        fill_mode='nearest',  # Заполнение пикселей вне границ ввода
        validation_split=0.1  # Указываем разделение изображений на обучающую и тестовую выборку
    )
    # обучающая выборка
    trainGenerator = datagen.flow_from_directory(
        imagePath,  # Путь ко всей выборке выборке
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # Размер изображений
        batch_size=batchSize,
        class_mode='categorical',  # Категориальный тип выборки. Разбиение выборки по маркам авто
        shuffle=True,  # Перемешивание выборки
        subset='training'  # устанавливаем как набор для обучения
    )

    # проверочная выборка
    validationGenerator = datagen.flow_from_directory(
        imagePath,  # Путь ко всей выборке выборке
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # Размер изображений
        batch_size=batchSize,
        class_mode='categoricl',  # Категориальный тип выборки. Разбиение выборки по маркам авто
        shuffle=True,  # Перемешивание выборки
        subset='validation'  # устанавливаем как валидационный набор
    )
    history = model.fit_generator(
        trainGenerator,
        steps_per_epoch=trainGenerator.samples // batchSize,
        validation_data=validationGenerator,
        validation_steps=validationGenerator.samples // batchSize,
        epochs=2,
        verbose=1
    )
    model.save('cars3.h5')
    model = load_model('cars3.h5', compile=False)
