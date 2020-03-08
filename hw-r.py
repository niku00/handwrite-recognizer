# github.com/niku00

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time

mnist = tf.keras.datasets.mnist  # 28x28 0-9 arası el yazılarının resimlerini yükledik.

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST datalarını değişkenlere atadık.

x_train = tf.keras.utils.normalize(x_train, axis=1)  # MNIST kütüphanelerinde alınan dataları 0-255 ten 0-1 arasına çevirdik.
x_test = tf.keras.utils.normalize(x_test, axis=1)  # Herhangi bir test görselini 0-255 ten 0-1 arasına çevirdik.

model = tf.keras.models.Sequential()  # Katman üretme stili seçildi. Sequential yada Functional seçilebilir.
model.add(tf.keras.layers.Flatten())  # Katmanların sıralı gösterilmesi için flatten komutu kullanıldı. 28x28 den 784x1 e geçildi.
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))  # Nöronları aktive edicek fonksiyon relu ve nöron sayısı 50 olarak belirlendi.
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))  # Nöronları aktive edicek fonksiyon relu ve nöron sayısı 50 olarak belirlendi. Deep learning katmanı.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Sonuçları 10 nörona indirdik. İhtimal hesabı yapmak için fonksiyonu softmax ile değiştirdik.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Modelin train biçimini belirledik.

model.fit(x_train, y_train, epochs=3)  # Modelin kaç tur training yapacağını belirledik

val_loss, val_acc = model.evaluate(x_test, y_test)  # Test sonuçlarını aldık.
print(val_loss, val_acc)  # Test sonuçlarını yazdırdık.

model.save('full.model')  # Modeli kaydettik.
new_model = tf.keras.models.load_model('full.model')  # Kaydedilen modeli açtık
predictions = new_model.predict(x_test)  # Modelin x_test sonucundaki tahminini bir değişkene atadık.

i = 1  # Test numarası değişkenini ve değerini belirledik.

plt.show()  # Plot ekranını açtık.

while (True):  # Sonsuz döngüye soktuk.
    random_number = random.randrange(0, 10000)  # 0-10000 arası rastgele sayı oluşturduk.

    prediction_number = np.argmax(predictions[random_number])  # Prediction tahminlerini rakama çevirdik.

    print("Test Number: {} , Prediction Result: {} , Example Number: {}".format(i, prediction_number, random_number))  # Tahminleri yazıya döktük.

    plt.imshow(x_test[random_number])  # x_test öğesinin gerçek resmini çizdirdik.
    plt.draw()  # Çizimi güncelledik.
    plt.pause(0.001)  # Plot ekranı için bekleme süresi ekledik.
    time.sleep(1)  # Çizimler arası bekleme süresi ekledik.
    i = i + 1  # Test numarasını yükselttik.
