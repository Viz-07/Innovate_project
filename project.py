import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Step 1: Data Collection (Using OpenCV's built-in dataset for simplicity)
# Load positive and negative samples
# Replace these with your dataset paths
pos_images = [cv2.imread(f'path_to_positive_samples/{i}.png') for i in range(1, 101)]
neg_images = [cv2.imread(f'path_to_negative_samples/{i}.png') for i in range(1, 101)]

# Data Augmentation Function
def augment_image(image):
    # Example augmentations: flipping and rotation
    flipped = cv2.flip(image, 1)  # Horizontal flip
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return [flipped, rotated]

# Augment positive images
augmented_pos_images = []
for img in pos_images:
    augmented_pos_images.extend(augment_image(img))

# Combine positive and negative samples
all_pos_images = pos_images + augmented_pos_images
all_neg_images = neg_images

# Shuffle the dataset
all_pos_images, all_neg_images = shuffle(all_pos_images, all_neg_images)

# Step 2: Preprocessing
def preprocess_image(image):
    resized_img = cv2.resize(image, (64, 128))
    normalized_img = resized_img / 255.0  # Normalize pixel values
    return normalized_img

# Extract preprocessed images
X_pos = np.array([preprocess_image(img) for img in all_pos_images])
X_neg = np.array([preprocess_image(img) for img in all_neg_images])

# Combine positive and negative samples
X = np.concatenate((X_pos, X_neg), axis=0)
y = np.array([1] * len(X_pos) + [0] * len(X_neg))

# Step 3: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model in batches
batch_size = 32
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Step 4: Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Step 5: Integration and Testing (Using the trained CNN model)
def detect_pedestrians(image):
    resized_img = cv2.resize(image, (64, 128))
    normalized_img = resized_img / 255.0  # Normalize pixel values
    prediction = model.predict(np.expand_dims(normalized_img, axis=0))
    if prediction[0][0] > 0.5:
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
        cv2.putText(image, "Pedestrian", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    return image

# Load a test image
test_image = cv2.imread('path_to_test_image/test_image.png')  # Add your path to a test image
detected_image = detect_pedestrians(test_image)

# Display the result
cv2.imshow('Pedestrian Detection', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
