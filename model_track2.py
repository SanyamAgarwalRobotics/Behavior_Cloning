import csv
import cv2
import sklearn
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, optimizers
from keras.layers import Flatten,Dense,Activation,Lambda,MaxPooling2D,Cropping2D,Convolution2D,Dropout,ELU
from keras.callbacks import ModelCheckpoint

'''

Open csv file from data directory, read each line and store them into sample_lines list
'''
sample_lines = []
with open('./track2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        sample_lines.append(line)
    #print(sample_lines) 

'''
Store left, center and middle images from each line and the corresponding
steering angles, an empirical correction of 0.2 is applied to steering
angle for left and right camera image.
'''
images       = []
measurements = []
count = 0
for line in sample_lines:
    #print("line : ",line)
    for i in range(3):
        source_path = line[i]
        #print(source_path)
        token = source_path.split('/')
        #print("token :",token)
        filename = token[-1]
        #print(filename)
        
        #find the image type left, right or center
        direction = filename.split("_")
        image_path = "./track2/IMG/" + filename
        #count = count + 1
        #print(image_path)
        
        #Read the image
        new_image = cv2.imread(image_path)
        
        '''
        The input image is split into YUV planes and passed to the network as recommended by
        NVIDIA, section 'Network Architecture'
        https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
        '''
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        #rows, cols, dim = new_image.shape

        images.append(new_image)
        
        measurement = float(line[3])
        
        '''
        Create adjusted steering measurements for the side camera images.
        these side camera images will be used to avoid bias towards left
        or right turn.
        '''
        correction = 0.2
        steering   = 0.0

        if "left" in direction:
            steering = measurement + correction
            #print("left steering ", steering)
        elif "right" in direction:
            steering = measurement - correction
            #print("right steering ", steering)
        else:
            steering = measurement
            #print("center steering ", steering)

        measurements.append(steering)
'''
split the data into 80% training and 20% validation
'''
train_samples, validation_samples,measurement_samples,measurement_validation_samples = train_test_split(images,measurements,test_size=0.2)

print("train_samples.shape", len(train_samples))
print("validation_samples.shape", len(validation_samples))
print("meaasurement_samples.shape", len(measurement_samples))
print("measurement_validation_samples.shape", len(measurement_validation_samples))
#print(rows)
#print(cols)
#print(dim)
#print("count : ",count)

def show_image(image):
    '''
    Show the image
    '''
    #print(name)
    plt.imshow(image)


def generator(samples,sample_measurements,batch_size=32):

    num_samples = len(samples)
    while(1): #Loop forever so the generator never terminates
        shuffle(samples,sample_measurements)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_measurements = sample_measurements[offset:offset+batch_size]
            augmented_images = []
            augmented_measurements = []

            #Iterate the image and steering angle on each batch
            for image, steering_angle in zip(batch_samples, batch_measurements):

                # Gausian Blur/down sample the original image to reduce noise
                image = cv2.GaussianBlur(image, (3,3), 0)
                augmented_images.append(image)
                augmented_measurements.append(steering_angle)

                #if steering angle is more than +- 43.0 flip the image
                # this is helpful to avoid bias towards left or right corner
                #if abs(steering_angle) > 0.03 and steering_angle < .30:
                      
                    # flip vertically (y-axis) and change the measurement sign
                flip_image = cv2.flip(image, 1)
                augmented_images.append(flip_image)
                augmented_measurements.append(steering_angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
batch_size = 64
total_epochs = 8

train_generator = generator(train_samples, measurement_samples, batch_size)
validation_generator = generator(validation_samples, measurement_validation_samples,  batch_size)

def model_mean_sqr_error_loss(model_fit_history):

    '''
    Draw mse verses epoch to find an optimum number of epoch
    '''
    # print the keys contained in history object
    print(model_fit_history.history.keys())

    #plot training and validation loss for each epoch
    plt.plot(model_fit_history.history['loss'])
    plt.plot(model_fit_history.history['val_loss'])
    plt.title('Model showing  mse loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


model = Sequential()

filter_size = 3
pool_size = (2,2)
'''
Make each pixel normalized and mean centered i.e. close to zero mean and equal variance which
is a good starting point for optimizing the loss to avoid too big or too small
'''
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))


'''
Crop the tree, hills, Sky from top 70 and the hood of the car from bottom 25 to avoid
noise
'''
model.add(Cropping2D(cropping=((70,25),(0,0))))


'''
Activation Layer selection
Based on the following recent experiment i.e.
https://ctjohnson.github.io/Capstone-Activation-Layers/ , section 5. Table of Results
I decided to stick with RELU as preferred activation layer to introduce nonlinearity
instead of Expotential Linear Unit (ELU)
'''
'''
model.add(Convolution2D(3,1,1,
                        border_mode='valid',
                        name='conv0', init='he_normal'))
model.add(Convolution2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv1', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv2', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv3', init='he_normal'))
model.add(ELU())

model.add(Convolution2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv4', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv5', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv6', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,name='hidden1', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(64,name='hidden2', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(16,name='hidden3',init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1, name='output', init='he_normal'))
'''
# 5x5 kernel with strides of 2x2, input depth 3 output depth 24
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 24 output depth 36
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 36 output depth 48
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 48 output depth 64
model.add(Convolution2D(64,3,3,activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 64 output depth 64
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

'''

'''
#model.add(Dropout(0.40))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.20))
model.add(Dense(10))
model.add(Activation('relu'))

'''

Ouput Directly predict the steering measurement, so 1 output
'''
model.add(Dense(1))

model.summary()
adam = optimizers.Adam(lr=0.001)

# checkpoint
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

'''
Compile and train the model using the generator function
'''
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, samples_per_epoch = \
                 len(train_samples), \
                 validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), \
                 nb_epoch=total_epochs, verbose = 1,callbacks=callbacks_list)

model.save('model.h5')

model_mean_sqr_error_loss(history_object)
