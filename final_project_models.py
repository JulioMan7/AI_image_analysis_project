import os
from keras.datasets import mnist
import random 
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Softmax, Multiply
import tensorflow as tf

#load the data from the mnist dataset
(train_img, train_label), (test_img, test_label) = mnist.load_data()

# Combine the features and labels for shuffling
train_data = list(zip(train_img, train_label))

# Shuffle the training data
np.random.shuffle(train_data)

# Unzip the shuffled data back into separate arrays for features and labels
train_img, train_label = zip(*train_data)

# Convert back to numpy arrays if needed
train_img = np.array(train_img)
train_label = np.array(train_label)

#split training data into train and validation sets
val_split = 1/6
val_size = int(val_split * len(train_img))

val_img = train_img[:val_size]
val_label = train_label[:val_size]

train_img = train_img[val_size:]
train_label = train_label[val_size:]

#normalize pixel values to range [0, 1]
train_img = train_img / 255.0
test_img = test_img / 255.0
val_img = val_img / 255.0

#visualize size of data sets
print(train_img.shape)
print(test_img.shape)
print(val_img.shape)
print(train_label.shape)
print(test_label.shape)
print(val_label.shape)

#visualize 10 images and their labels from each set
def plot_images(images, labels, title):
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Plot images and labels from the training set
plot_images(train_img[:10], train_label[:10], "Training Set")

# Plot images and labels from the testing set
plot_images(test_img[:10], test_label[:10], "Testing Set")

# Plot images and labels from the testing set
plot_images(val_img[:10], val_label[:10], "Validation Set")

#one-hot ecode labels
train_label=to_categorical(train_label)
test_label=to_categorical(test_label)
val_label=to_categorical(val_label)

#create the models 
def create_baseline_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ])

def create_tanh_activation_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='tanh', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='tanh'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ])

# Define the custom attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.filters = input_shape[3]
        self.W = self.add_weight(name='attention_W', shape=(1, 1, self.filters, self.filters), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name='attention_b', shape=(self.filters,), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_weights = tf.nn.tanh(tf.nn.conv2d(inputs, self.W, strides=[1, 1, 1, 1], padding="SAME") + self.b)
        attention_weights = Softmax(axis=-1)(attention_weights)
        attended_inputs = Multiply()([inputs, attention_weights])
        return attended_inputs

def create_attention_mechanism_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        AttentionLayer(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ])


model_creation_functions = {
    "baseline_model": create_baseline_model,
    "tanh_activation_model": create_tanh_activation_model,
    "attention_mechanism_model": create_attention_mechanism_model
}

for model_name, create_model_func in model_creation_functions.items():
    model = create_model_func()  # Create a new instance of the model
    print(f"Model: {model_name}")
    model.summary()
    print("\n")

def train_and_save_model(create_model_func, train_img, train_label, test_img, test_label, model_name, num_epochs=9, batch_size=256):
    model = create_model_func()  # Create a new instance of the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_img, train_label, epochs=num_epochs, batch_size=batch_size, validation_data=(test_img, test_label))
    model.save(f'{model_name}.h5')
    return history

# Train and save all models
histories = {}
for model_name, create_model_func in model_creation_functions.items():
    for i in range(1, 4):
        history = train_and_save_model(create_model_func, train_img, train_label, test_img, test_label, f'{model_name}_{i}')
        histories[f'{model_name}_{i}'] = history
        print(f'{model_name.capitalize()} {i} trained and saved.')

# Display all histories with pyplot
for model_name, history in histories.items():
    plt.figure(figsize=(12, 4))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

######################
#   Testing Models   #
######################
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

loaded_models = {}
for model_name in model_creation_functions.keys():
    for i in range(1, 4):
        model_path = f'{model_name}_{i}.h5'
        if model_name == "attention_mechanism_model":
            loaded_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        else:
            loaded_model = load_model(model_path)
        loaded_models[f'{model_name}_{i}'] = loaded_model

for model_name, loaded_model in loaded_models.items():
    val_loss, val_acc = loaded_model.evaluate(val_img, val_label)
    print(f"{model_name} - Validation Accuracy: {val_acc}")

# Define the number of examples to choose
num_examples = 5

# Iterate through each loaded model
for model_name, loaded_model in loaded_models.items():
    print(f"Model: {model_name}")
    
    # Choose random indices from the validation set
    random_indices = np.random.choice(val_img.shape[0], num_examples)
    
    # Predict labels for the chosen examples
    predicted_labels = loaded_model.predict(val_img[random_indices])
    predicted_classes = np.argmax(predicted_labels, axis=1)
    
    # Plot the examples
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(val_img[idx].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Actual: {np.argmax(val_label[idx])}\nPredicted: {predicted_classes[i]}")
        plt.axis('off')
    plt.show()


# Create dictionaries to store predicted and true classes for all models
predicted_classes_all = {}
true_classes_all = {}

# Iterate through each loaded model
for model_name, loaded_model in loaded_models.items():
    print(f"Model: {model_name}")
    
    # Predict labels for all examples in the validation set
    predicted_labels_all = loaded_model.predict(val_img)
    predicted_classes = np.argmax(predicted_labels_all, axis=1)
    predicted_classes_all[model_name] = predicted_classes
    
    # Convert one-hot encoded labels back to integers
    true_classes = np.argmax(val_label, axis=1)
    true_classes_all[model_name] = true_classes


# Iterate through each loaded model
for model_name in loaded_models.keys():
    print(f"Model: {model_name}")
    
    # Find indices of incorrectly classified examples
    incorrect_indices = np.where(predicted_classes_all[model_name] != true_classes_all[model_name])[0]

    # Choose random incorrect indices
    num_incorrect_examples = 5
    random_incorrect_indices = np.random.choice(incorrect_indices, min(num_incorrect_examples, len(incorrect_indices)), replace=False)

    # Plot incorrect examples
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_incorrect_indices):
        plt.subplot(1, min(num_incorrect_examples, len(incorrect_indices)), i + 1)
        plt.imshow(val_img[idx].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Actual: {true_classes_all[model_name][idx]}\nPredicted: {predicted_classes_all[model_name][idx]}")
        plt.axis('off')
    plt.show()


# Iterate through each loaded model to create and plot confusion matrices
for model_name in loaded_models.keys():
    print(f"Model: {model_name}")
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(true_classes_all[model_name], predicted_classes_all[model_name])

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def most_confused_numbers(conf_matrix):
    """
    Find the most confused numbers based on the confusion matrix.
    
    Parameters:
        conf_matrix (numpy.ndarray): Confusion matrix.
    
    Returns:
        tuple: Tuple containing the most confused numbers.
    """
    max_confusion = -1
    most_confused = (-1, -1)

    # Iterate over each element of the confusion matrix
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            if i != j and conf_matrix[i][j] > max_confusion:
                max_confusion = conf_matrix[i][j]
                most_confused = (i, j)

    return most_confused

# Iterate through each loaded model to find the most confused numbers
for model_name in loaded_models.keys():
    print(f"Model: {model_name}")

    # Create confusion matrix
    conf_matrix = confusion_matrix(true_classes_all[model_name], predicted_classes_all[model_name])

    # Find the most confused numbers
    most_confused = most_confused_numbers(conf_matrix)
    print(f"Most Confused Numbers: {most_confused}")
