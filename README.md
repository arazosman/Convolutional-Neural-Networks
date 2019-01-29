# Convolutional-Neural-Networks
An image classifier program based on CNN.

1. Ön Hazırlık:

Ödevde nöral ağlar için Keras ve Tensorflow kütüphaneleri kullanıldı. Tensorflow 
kütüphanesi Python 3.7.x sürümlerini desteklemediği için bilgisayarınızda Python 
3.6.x sürümlerinden birisinin kurulu olması gerekiyor. Python indirmek için:
https://www.python.org/downloads/

Gerekli kütüphane kurulumları için terminal üzerinden aşağıdaki komutları çalıştırın:

pip install numpy --user
pip install tensorflow --user
pip install theano --user
pip install keras --user

---------------------------------------------------

2. Veri Setlerinin Oluşturulması:

Bize verilen veri setinin %80'lik kısmını eğitim için, %20'lik kısmını da test için 
kullandık. Bunun için "test" ve "training" olmak üzere iki klasör oluşturup resimleri 
klasörlere kategorilerine göre dengeli olarak dağıttık. Buna ek olarak kullanıcıların
kendi resimlerini, kurulan sistem üzerinde test edebilmesi için "predict" isimli bir
klasör daha oluşturduk. Sistem, bu klasör içindeki resimleri, oluşturduğu nöral ağlar
üzerinde test ederek ait oldukları sınıfları tahmin ediyor.

---------------------------------------------------

3. Programın Çalışması

Program, eğitim resimlerinin bulunduğu klasör, test resimlerinin bulunduğu klasör ve 
sınıfları tahmin edilecek resimlerinin bulunduğu klasörün konumlarını kullanıcıdan alır.
(Eğitim ve test klasörleri mutlaka bulunmalıdır (içerikleri boş olabilir). Tahmin için
kullanılacak klasör ise opsiyoneldir.) Programla beraber indirdiğiniz klasörleri varsayılan
klasörler olarak kullanabilirsiniz.

Klasörler belirlendikten sonra eğitim aşamasına geçilir. Kullanıcıdan alınan devir (epoch)
sayısı kadar çeşitli tabakalar uygulanır. Convolution tabakası ile resimlerin karakteristik 
özellikleri çıkarılır. RELU tabakası ile convolution işlemi sonrası oluşan matristeki negatif
değerler 0 yapılır. Pooling tabakası ile oluşan matris boyutu düşürülür. Flatten tabakası
ile pooling matrisleri birleştirilerek tek boyutlu bir dizi elde edilir. Ve nihayet çeşitli 
Dense tabakaları ile resimlere sınıflarına göre olasılık değerleri atanır.

Eğitim aşamasına verilen test resimleriyle oluşturulan sistemin başarısı ölçülür. Ayrıca
kullanıcının sisteme tahmin için resimler vermesi durumunda sistem, verilen her bir resim 
için sınıf tahmininde bulunur.
