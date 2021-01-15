# Imports
import numpy
import os.path
import itertools
import tensorflow as tf
from random import randint
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model, model_from_json

# Initialize data
X_train = []
y_train = []
X_test = []
y_test = []

# Example data: 
# - An experiemental drug was tested on individuals from ages 13 to 100 in a clinical trial
# - The trial had 2100 participants. Half were under 65 years old, half were 65 years or older
# - Around 95% of patients 65 or older experienced side effects
# - Around 95% of patients under 65 experienced no side effects

# Train data
for i in range(50):
		# The 5% of younger individuals who did experience side effects
		random_younger = randint(13, 64)
		X_train.append(random_younger)
		y_train.append(1)
		
		# The 5% of older individuals who did not experience side effects
		random_older = randint(65, 100)
		X_train.append(random_older)
		y_train.append(0)

for i in range(1000):
		# The 95% of younger individuals who did not experience side effects
		random_younger = randint(13, 64)
		X_train.append(random_younger)
		y_train.append(0)
		
		# The 95% of older individuals who did experience side effects
		random_older = randint(65, 100)
		X_train.append(random_older)
		y_train.append(1)

# Test data
for i in range(10):
		# The 5% of younger individuals who did experience side effects
		random_younger = randint(13, 64)
		X_test.append(random_younger)
		y_test.append(1)
		
		# The 5% of older individuals who did not experience side effects
		random_older = randint(65, 100)
		X_test.append(random_older)
		y_test.append(0)

for i in range(200):
		# The 95% of younger individuals who did not experience side effects
		random_younger = randint(13, 64)
		X_test.append(random_younger)
		y_test.append(0)
		
		# The 95% of older individuals who did experience side effects
		random_older = randint(65, 100)
		X_test.append(random_older)
		y_test.append(1)


# Convert to the format numpy array
X_train = numpy.array(X_train)
y_train = numpy.array(y_train)
X_test = numpy.array(X_test)
y_test = numpy.array(y_test)

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Transform features by scaling each feature to a given range
minMaxScaler = MinMaxScaler(feature_range = (0, 1))
X_train = minMaxScaler.fit_transform(X_train.reshape(-1, 1))
X_test = minMaxScaler.fit_transform(X_test.reshape(-1, 1))

# Print shape of the data
print("Train samples shape:", X_train.shape)
print("Train labels shape:", y_train.shape)

# Check the number of available GPUs
physical_devices = tf.config.experimental.list_physical_devices("GPU")
print("Number of GPUs Available:", len(physical_devices))
# Set if memory growth should be enabled for a PhysicalDevice
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize model
model = Sequential([Dense(units = 16, input_shape = (1, ), activation = "relu"),
					Dense(units = 32, activation = "relu"),
					Dense(units = 2, activation = "softmax")])

# Summary of the model
model.summary()

# Compile the model
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fit the model
model.fit(x = X_train, y = y_train, validation_split = 0.1, batch_size = 10, epochs = 30, shuffle = True, verbose = 2)

# Predict using the model
y_pred = model.predict(x = X_test, batch_size = 10, verbose = 0)

# Convert to rounded prediction
y_pred = numpy.argmax(y_pred, axis = -1)

# Confusion matrix
cm = confusion_matrix(y_true = y_test, y_pred = y_pred)

# Fonction to plot confusion matrix
def plot_confusion_matrix(cm, cm_plot_labels, normalize, title, cmap):
	plt.imshow(cm, interpolation = "nearest", cmap = cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = numpy.arange(len(cm_plot_labels))
	plt.xticks(tick_marks, cm_plot_labels, rotation = 45)
	plt.yticks(tick_marks, cm_plot_labels)

	if normalize:
		cm = cm.astype("float") / cm.sum(axis = 1)[:, numpy.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if i == 0: plt.text(j, i + 0.25, cm[i, j], horizontalalignment = "left", verticalalignment = "baseline", color = "black" )
		else: plt.text(j, i - 0.25, cm[i, j], horizontalalignment = "left", verticalalignment = "baseline", color = "black")
	plt.tight_layout()
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()

# Set the labels
cm_plot_labels = ["No_Side_Effects", "Had_Side_Effects"]
# Plot the confusion matrix
plot_confusion_matrix(cm = cm, cm_plot_labels = cm_plot_labels, normalize = False, title = "Confusion matrix", cmap = plt.cm.Blues)

# Save model
if os.path.isfile("models/medical_trial_model.h5") is False:
	model.save("models/medical_trial_model.h5")

# Load model
new_model = load_model("models/medical_trial_model.h5")

# Summary of the model
new_model.summary()

# Get weights of the model
weights = new_model.get_weights()

# Get optimizer
optimizer = new_model.optimizer

# Save architecture of the model as JSON or as YAML
json = model.to_json()
yaml = model.to_yaml()

# Model reconstruction from JSON or from YAML
model_architecture_from_json = model_from_json(json)
model_architecture_from_yaml = model_from_yaml(yaml)