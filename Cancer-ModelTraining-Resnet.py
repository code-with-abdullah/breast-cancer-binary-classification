# -*- coding: utf-8 -*-


import os 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50

base_dir = os.getcwd()

train_dir = os.path.join(base_dir, "Data", 'train')
validation_dir = os.path.join(base_dir, "Data", 'validation')
test_dir = os.path.join(base_dir, "Data", 'test')

train_datagen = ImageDataGenerator( rescale = 1.0/255. )
validation_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (50, 50))
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size = 20, class_mode = 'binary', target_size = (50, 50))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 20, class_mode = 'binary', target_size = (50, 50))

base_model = ResNet50(input_shape = (50, 50, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')


# feature extraction - image net weights (transfer learnt)
# classification - last layer (trained on custom dataset)

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001), loss = 'binary_crossentropy', metrics = ['acc'])


print(model.summary())


resnet_history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 15)


if not (os.path.exists("saved weights_ResNet50")):
    os.mkdir("saved weights_ResNet50")
    
    
def save_model_to_file(model, iteration_number, percentage_achieved):
    model.save(f"saved weights_ResNet50/model_{iteration_number}.h5")
    
    with open("saved weights_ResNet50/log.txt", "a+") as logger:
        logger.write(str(percentage_achieved) + "\n")

    print("Model saved successfully")
    
    
    
    
save_model_to_file(model, 1, "71")



vgghist2 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 5)

save_model_to_file(model, 2, "74")

vgghist2 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 5)

save_model_to_file(model, 2, "74")




import numpy as np
from PIL import Image
from skimage import transform



def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255.
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


negative_misprediction = 0
for image in os.listdir(os.path.join(test_dir, "0")):
    prediction = model.predict(load(os.path.join(test_dir, "0", image)))
    if prediction >= 0.5:
        negative_misprediction = negative_misprediction + 1
        print("Incorrect prediction")
    else:
        print("Correct prediction")
        
        
        
total_negative_test_images = len(os.listdir(os.path.join(test_dir, "0")))
print("Total images in negative validation set are ", total_negative_test_images)
print("Total correct classifications are ", total_negative_test_images - negative_misprediction)
print("Total incorrect classifications are ", negative_misprediction)


positive_misprediction = 0
for image in os.listdir(os.path.join(test_dir, "1")):
    prediction = model.predict(load(os.path.join(test_dir, "1", image)))
    if prediction < 0.5:
        positive_misprediction = positive_misprediction + 1
        print("Incorrect prediction")
    else:
        print("Correct prediction")
        
        
        
total_positive_test_images = len(os.listdir(os.path.join(test_dir, "1")))
print("Total images in negative validation set are ", total_positive_test_images)
print("Total correct classifications are ", total_positive_test_images - positive_misprediction)
print("Total incorrect classifications are ", positive_misprediction)



total_test_images = total_positive_test_images + total_negative_test_images
total_correct_classifications = (total_positive_test_images - positive_misprediction) + (total_negative_test_images - negative_misprediction)
total_incorrect_classifications = positive_misprediction + negative_misprediction

print("Total test images ", total_test_images)
print("Correctly classified test images ", total_correct_classifications)
print("Incorrectly classified test images ", total_incorrect_classifications)

print(f"Accuracy Percentage ({total_correct_classifications } / {total_test_images}) * 100 = ",
      (total_correct_classifications/total_test_images)*100)





