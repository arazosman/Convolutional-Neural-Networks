'''
    Görüntü İşleme - Dönem Projesi
    Evrişimsel Sinir Ağları

    Hazırlayanlar:
    Kıymet Çelebi - 15011073
    Osman Araz - 16011020

    Teslim Tarihi: 02.01.2019
'''

'''
    Ödevde nöral ağlar için Keras ve Tensorflow kütüphaneleri kullanıldı. Tensorflow 
    kütüphanesi Python 3.7.x sürümlerini desteklemediği için bilgisayarınızda Python 
    3.6.x sürümlerinden birisinin kurulu olması gerekiyor. Python indirmek için:
    https://www.python.org/downloads/

    Gerekli kütüphane kurulumları için terminal üzerinden aşağıdaki komutları çalıştırın:
    
    pip install numpy --user
    pip install tensorflow --user
    pip install theano --user
    pip install keras --user
'''

import os
import sys
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.image as mpimg

def getCategories(pathOfTrainingImages):
    '''
    Resim sınıflarını bulan fonksiyon.
    '''
    
    categories = []

    pathOfCategories = os.listdir(pathOfTrainingImages)

    for path in pathOfCategories:
        categories.append(path)

    return categories

def predictForImages(CNN, categories, predictPath, imageSize):
    '''
    Kullanıcı tarafından verilen resimlerin sınıflarını tahmin eden fonksiyon.
    '''

    print("\nTahminler:\n")

    listOfImages = os.listdir(predictPath)

    for imageName in listOfImages:
        path = os.path.join(predictPath, imageName)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imageSize, imageSize))
        image = np.array(image).reshape(-1, imageSize, imageSize, 3)
        preds = CNN.predict(image)
        indice = preds.argmax(axis = -1)[0]
        print(imageName, "->", categories[indice])

def trainingDataset(CNN, epochs, imageSize, categories, pathOfTrainingImages, pathOfTestImages):
    '''
    Verilen eğitim resimlerini kullanarak nöral ağları oluşturan ve test resimleriyle
    başarı yüzdesini ölçen fonksiyon. 
    '''
    
    # İlk convolution tabakası:
    # 32 tane filtre 3x3 boyutunda filtre uygulanıyor.
    # Convolution işleminin ardından RElU işlemi uygulanıyor. (Negatif değerler 0'a eşitlenir.)
    # Resim, 32x32 boyutunda ve RGB formatında ayarlanıyor.
    CNN.add(Conv2D(filters = 64, 
                kernel_size = (3, 3), 
                activation = 'relu',
                input_shape = (imageSize, imageSize, 3)))

    # Pooling işlemi:
    # 2x2 boyutunda, x ve y koordinatlarında 2'şer birim atlayan pooling uygulanıyor.
    CNN.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Bir convolution tabakası daha uygulanıyor:
    # 64 tane filtre 3x3 boyutunda filtre uygulanıyor ve ardından RElU işlemi uygulanıyor.
    CNN.add(Conv2D(filters = 64, 
                    kernel_size = (3, 3), 
                    activation = 'relu'))

    # 2x2 boyutunda, x ve y koordinatlarında 2'şer birim atlayan pooling uygulanıyor.
    CNN.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Resimler tek boyuta indirgeniyor:
    CNN.add(Flatten())

    # Fully-Connected tabakası uygulanıyor:
    CNN.add(Dense(128, activation = 'relu'))
    CNN.add(Dense(128, activation = 'sigmoid'))

    # Softmax akstivasyonuyla kategorilere olasılık değerleri atanıyor:
    CNN.add(Dense(len(categories), activation = 'softmax'))

    # Oluşturulan CNN derleniyor:
    CNN.compile(loss = "sparse_categorical_crossentropy", 
                optimizer = 'adam', 
                metrics = ['accuracy'])

    # Eğitim resimlerini saklamak için bir veri yapısı oluşturuluyor.
    # rescale parametresiyle resimler normalize edilecek.
    trainingDataGenerator = ImageDataGenerator(rescale = 1/255)

    # Eğitim resimleri için oluşturulan veri yapısı için resimler bilgisayardan alınıyor.
    # Resimler 32x32 boyutunda RGB formatında okunuyor.
    # Resimler kategorilere ayrılıyor.
    trainingImages = trainingDataGenerator.flow_from_directory(pathOfTrainingImages,
                                                        target_size = (imageSize, imageSize),
                                                        color_mode = 'rgb',
                                                        class_mode = 'binary',
                                                        classes = categories)

    # Test resimlerini saklamak için bir veri yapısı oluşturuluyor.
    # Değerler normalize edilecek.
    testDataGenerator = ImageDataGenerator(rescale = 1/255)

    # Test resimleri için oluşturulan veri yapısı için resimler bilgisayardan alınıyor.
    # Resimler 32x32 boyutunda RGB formatında okunuyor.
    # Resimler kategorilere ayrılıyor.
    testImages = testDataGenerator.flow_from_directory(pathOfTestImages,
                                                    target_size = (imageSize, imageSize),
                                                    color_mode = 'rgb',
                                                    class_mode = 'binary',
                                                    classes = categories)

    numOfTrainingImages = sum(len(files) for _, _, files in os.walk(pathOfTrainingImages))
    numOfTestImages = sum(len(files) for _, _, files in os.walk(pathOfTestImages))

    # Eğitim işlemi 5 devirde uygulanıyor:
    CNN.fit_generator(trainingImages,
                    epochs = epochs,
                    steps_per_epoch = numOfTrainingImages,  
                    validation_steps = numOfTestImages,
                    validation_data = testImages)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #  'AVX2' uyarısından kurtulmak için

    imageSize = 32

    print("\nEğitim resimlerinin konumunu girin:")
    print("(Varsayılan konum 'Dataset/training' için boş bırakın):", end = " ")

    trainingPath = input()

    if (len(trainingPath) == 0):
        trainingPath = "Dataset/training"

    if (os.path.exists(trainingPath) == False):
        print("Hatalı konum.")
        sys.exit()

    print("\nTest resimlerinin konumunu girin:")
    print("(Varsayılan konum 'Dataset/test' için boş bırakın):", end = " ")

    testPath = input()

    if (len(testPath) == 0):
        testPath = "Dataset/test"

    if (os.path.exists(testPath) == False):
        print("Hatalı konum.")
        sys.exit()

    categories = getCategories(trainingPath)

    epochs = int(input("\nDevir (epoch) sayısını girin: "))

    print("\nEğitim işlemi başlatılıyor. Eğitim bittikten sonra vereceğiniz resimlerle tahmin işlemi yapabileceksiniz.")
    print("Lütfen işlemin bitmesini bekleyin...\n")

    # CNN modeli oluşturuluyor, bütün işlemler bunun üzerinden gerçekleştirilecek:
    CNN = Sequential()

    trainingDataset(CNN, epochs, imageSize, categories, trainingPath, testPath)

    print("\nEğitim işlemi tamamlandı.")
    
    choice = input("Vereceğiniz resimlere tahmin uygulatmak istiyor musunuz? (E/H): ")

    if choice.upper() == "E":
        print("\nTahmin uygulanacak resimlerin konumunu girin:")
        print("(Varsayılan konum 'Dataset/predict' için boş bırakın):", end = " ")

        predictPath = input()

        if (len(predictPath) == 0):
            predictPath = "Dataset/predict"

        if (os.path.exists(predictPath) == False):
            print("Hatalı konum.")
            sys.exit()

        predictForImages(CNN, categories, predictPath, imageSize)

if __name__ == "__main__":
    main()