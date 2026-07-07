# model_handler.py
import numpy as np
import tensorflow as tf
from PIL import Image

def load_mnist_model():
    try:
        # Aapka train kiya hua model load karega
        model = tf.keras.models.load_model('mnist_model.h5')
        print("Model successfully load ho gaya!")
        return model
    except Exception:
        # Agar h5 file missing ho toh test karne ke liye automated model bana dega
        print("mnist_model.h5 nahi mili. Ek dummy model create ho raha hai...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

def preprocess_image(image: Image.Image):
    # Image ko grayscale (L) mein convert karna aur 28x28 size karna
    image = image.convert('L').resize((28, 28))
    img_array = np.array(image)
    
    # Pixel values ko 0-1 ke darmiyan normalize karna
    img_array = img_array / 255.0
    
    # Batch dimension add karna (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array