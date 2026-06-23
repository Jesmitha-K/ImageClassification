from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("furniture_cnn.h5")

img = image.load_img("captured.jpg", target_size=(224,224))
img = np.expand_dims(img, axis=0)
img = img / 255.0

pred = model.predict(img)

classes = ['Almirah', 'Chair', 'Refrigerator', 'Table', 'Television']

print("Prediction:", classes[np.argmax(pred)])
print("Confidence:", np.max(pred))