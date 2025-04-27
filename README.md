# Banana Quality Classification

> Proyek klasifikasi kualitas pisang menggunakan Convolutional Neural Network (CNN) dengan transfer learning di Google Colab.

## 📋 Deskripsi

Proyek ini bertujuan mengenali kualitas pisang (overripe, ripe, rotten, unripe) menggunakan dataset berisi lebih dari 10.000 gambar. Model dibangun dengan **MobileNetV2** (transfer learning) dan dilatih dalam dua tahap (feature extraction & fine‑tuning). Hasil disimpan dalam format SavedModel, TFLite, dan TFJS.

## 🚀 Fitur Utama

- Split dataset otomatis (70% train, 15% val, 15% test)
- Augmentasi lanjutan: rotasi, shift, zoom, brightness, flip
- Transfer learning dengan MobileNetV2 + fine‑tuning
- Callback: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
- Visualisasi: plot accuracy & loss, confusion matrix
- Evaluasi: akurasi ≥95%, classification report
- Penyimpanan model: SavedModel, TFLite (`.tflite`), TFJS
- File `label.txt` untuk mapping TFLite
- Manual inference baik via Keras maupun TFLite

## 📦 Persyaratan

- Python 3.8+  
- TensorFlow 2.13+  
- TensorFlow.js converter  
- scikit-learn (untuk classification report, confusion matrix)  
- Google Colab (opsional)

```bash
pip install tensorflow tensorflowjs scikit-learn matplotlib
```

## 🗂 Struktur Proyek

```
submission
├───data_splitted
| ├───ripe
| ├───unripe
| ├───rotten
| └───overripe
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───best_model.h5
├───notebook.ipynb
├───README.md
└───requirements.txt

```

## 📖 Cara Menjalankan

1. **Clone atau salin** ke Google Colab.
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Ekstrak** dataset (ZIP) ke direktori.
4. **Instal** dependencies:
   ```bash
   !pip install tensorflowjs
   pip install tensorflow scikit-learn matplotlib
   ```
5. **Jalankan** cell per langkah:
   - Setup & import library
   - Pengecekan dataset & split
   - Data generator & augmentasi
   - Build model (feature extraction)
   - Fine‑tuning
   - Visualisasi metrics
   - Evaluasi & confusion matrix
   - Simpan model
   - Manual inference

## 🎯 Hasil

- **Test Accuracy**: ≥95% (model.evaluate)
- **Batch inference**: ~96%
- **Classification Report**: precision, recall, F1-score per kelas
- **Confusion Matrix**: visualisasi kesalahan prediksi

## 🔍 Manual Inference

Gunakan snippet berikut di Colab setelah model tersedia:

```python
from google.colab import files
import numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing import image

# Keras
model = tf.keras.models.load_model('best_model.h5')
labels = [ ... ]  # auto_labels
uploaded = files.upload()
for fname in uploaded:
    # preprocess & predict
    
# TFLite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
...
```

## 📁 Output

- `label.txt`: daftar kelas
- `best_model.h5`: bobot model terbaik
- `model.tflite`: model untuk perangkat mobile
- `saved_model/`: format TensorFlow
- `tfjs_model/`: untuk browser (TensorFlow.js)

## 📜 Lisensi

MIT License. Silakan modifikasi sesuai kebutuhan.

