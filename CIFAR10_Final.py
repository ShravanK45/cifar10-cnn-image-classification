from altair import layer
from sympy import rotations
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#Normlaize Pixel Values to range [0,1]
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#One-Hot Encode the labels
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

#Data Augmentation
datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

#Fit the gen to training data
datagen.fit(x_train)

#CNN Model
def create_model():
    model= models.Sequential()

    #Convulational Layer1
    model.add(layers.Input(shape=(32,32,3)))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(0.3))

    #Convulational Layer2
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(0.3))


    #Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation='softmax'))

    return model


model=create_model()
model.summary()

model.compile(
    optimizer='adam',
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------

# Stops training when validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Save only the best model
checkpoint = ModelCheckpoint(
    "best_cifar10_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

#Train the model using the augmented data generator
history= model.fit(
     datagen.flow(x_train,y_train,batch_size=64),
    epochs=30,
    validation_data=(x_test,y_test),
    steps_per_epoch=x_train.shape[0] // 64,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)

test_loss,test_accuracy=model.evaluate(
    x_test,y_test,
    verbose=2
)

print(f"Test Accuracy is {test_accuracy}")

#Visualise the accuracy and loss
plt.plot(history.history['accuracy'],label="Training Accuracy")
plt.plot(history.history['val_accuracy'],label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'],label="Training Loss")
plt.plot(history.history['val_loss'],label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
