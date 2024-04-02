import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import my_config
from my_config import BATCH_SIZE, LOGMEL_SHAPE_WINDOW
from data_generator import DataGenerator


test_generator = DataGenerator(
    my_config.TEST_H5,
    batch_size=BATCH_SIZE,
    audio_shape=LOGMEL_SHAPE_WINDOW,
    verbose=False
)


def test_gen():
    labels_list = []
    for features, labels in test_generator:
        labels_list.extend(labels)
        yield features, labels
    return labels_list


# Adjust the output_signature according to your actual data shape and types
test_dataset = tf.data.Dataset.from_generator(
    test_gen,
    output_signature=(
        {
            "input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE_WINDOW, dtype=tf.float32),  # Adjust shape
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust shape for your labels
    ),
)

# Load true labels from test generator
test_labels = []
for _, labels in test_generator:
    test_labels.extend(labels)

# Load the model if not already in memory
model = keras.models.load_model('./model/first_model_SAD.keras')

# Make predictions on the test set using the generator
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Ensure the length of test_labels matches the number of predictions
assert len(test_labels) == len(predicted_classes), "Mismatch in number of true and predicted labels."

# Compute the confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['Non-depressed', 'Depressed'],
    yticklabels=['Non-depressed', 'Depressed']
)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(test_labels, predicted_classes, target_names=['Non-depressed', 'Depressed']))
