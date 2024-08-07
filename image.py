import keras
import keras, os
from keras.models import load_model

def job():
    dir_list = os.listdir("media/Pictures")
    for file_name in dir_list:
        os.rename(f'media/Pictures/{file_name}' , 'media/Pictures/image.jpg')


model = load_model('app_name/saved_model.hdf5')

def imageIdentification(file_input):
    image_size = (180, 180)
    img = keras.utils.load_img(file_input, target_size=image_size)
    # import numpy
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    return f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog."


if __name__=='__main__':
    imageIdentification('')
