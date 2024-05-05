import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\Cong\\Desktop\\football player train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',)

validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\Cong\\Desktop\\football player test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5),  padding = 'same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),  padding = 'same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides = (2,2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),  padding = 'same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(24, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(train_generator,
          epochs=50,
          validation_data=validation_generator)




model.save('foodballplayer.h5')