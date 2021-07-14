import os
from src import config

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#                      METHOD 1
# ==================================================== #
#             Using dataset_from_directory             #
# ==================================================== #

class DatasetFromDirectory(datasets):
    def __init__(self, 
                path, 
                validation_split=0.0,
                label_mode="categorical", 
                color_mode="rgb", 
                shuffle=True
            ):
        super(DatasetFromDirectory, self).__init__()
        self.path = path
        self.label_mode = label_mode
        self.validation_split = validation_split

    def augmentation(self, image):
        image = tf.image.random_brightness(image, max_delta=0.05)
        return image

    def load(self):
        ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            labels="inferred",
            label_mode=self.label_mode,  # categorical, binary, int
            # class_names=['0', '1', '2', '3', ...]
            color_mode=self.color_mode,
            batch_size=config.BATCH_SIZE,
            image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),  # reshape if not in this size
            shuffle=self.shuffle,
            seed=123,
            validation_split=self.validation_split,
            subset="training",
        )

        ds_train = ds_train.map(self.augmentation())

        if self.validation_split:
            ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
                self.path,
                labels="inferred",
                label_mode=self.label_mode,  # categorical, binary
                # class_names=['0', '1', '2', '3', ...]
                color_mode=self.color_mode,
                batch_size=config.BATCH_SIZE,
                image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),  # reshape if not in this size
                shuffle=self.shuffle,
                seed=123,
                validation_split=self.validation_split,
                subset="validation",
            )

        return ds_train, ds_validation




#                           METHOD 2
# ================================================================== #
#             ImageDataGenerator and flow_from_directory             #
# ================================================================== #

    class ImageDatagenerator(datasets):
        def __init__(self,
                    path,
                    validation_split=0.0,
                    label_mode="categorical",
                    color_mode="rgb",
                    class_mode="",
                    channels_last=True,
                    shuffle=True
                ):
            super(DatasetFromDirectory, self).__init__()
            self.path = path
            self.label_mode = label_mode
            self.color_mode = color_mode
            self.class_mode = class_mode # sparse, categorical, binary, multi-input
            self.channels_last = channels_last
            self.validation_split = validation_split

        def load(self):
            datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=5,
                zoom_range=(0.95, 0.95),
                horizontal_flip=False,
                vertical_flip=False,
                data_format="channels_last" if self.channels_last else "channels_first",
                validation_split=self.validation_split,
                dtype=tf.float32,
            )

            train_generator = datagen.flow_from_directory(
                self.path,
                target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                batch_size=config.BATCH_SIZE,
                color_mode=self.color_mode,
                class_mode=self.class_mode,
                shuffle=self.shuffle,
                subset="training",
                seed=123,
            )

            if validation_split:
                valid_generator = datagen.flow_from_directory(
                self.path,
                target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                batch_size=config.BATCH_SIZE,
                color_mode=self.color_mode,
                class_mode=self.class_mode,
                shuffle=self.shuffle,
                subset="validation",
                seed=123,
                )

            return train_generator, valid_generator



#                           METHOD 3
# ================================================================== #
#                            FromCSV                                 #
# ================================================================== #

directory = "data/mnist_images_csv/"
df = pd.read_csv(directory + "train.csv")

file_paths = df["file_name"].values
labels = df["label"].values
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))


def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label


def augment(image, label):
    # data augmentation here
    return image, label
