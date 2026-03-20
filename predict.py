import numpy as np
import cv2
import argparse
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("food_model.h5")

IMG_SIZE = 224

# =========================
# SAMPLE CALORIE DATA
# =========================
calorie_dict = {
    "pizza": 285,
    "burger": 295,
    "salad": 150,
    "cake": 350,
    "fries": 312
}

# =========================
# LOAD CLASS NAMES
# =========================
import os
class_names = os.listdir("food-101/images")

# =========================
# ARGUMENT PARSER
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

# =========================
# IMAGE PROCESSING
# =========================
img = cv2.imread(args.image)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# =========================
# PREDICTION
# =========================
pred = model.predict(img)
class_id = np.argmax(pred)
food = class_names[class_id]

calories = calorie_dict.get(food, "Unknown")

# =========================
# OUTPUT
# =========================
print("🍽️ Food:", food)
print("🔥 Estimated Calories:", calories)
