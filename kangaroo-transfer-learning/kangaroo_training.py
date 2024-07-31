import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

#import tensorflow as tf

#For this script to run on the GPU, you must install CUDA, CUDANN, and tensorflow -gpu(it has a tensorflow backend)
#can still run on cpu but it will take about 1 hr per epoch :(

class ShellfishDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "Healthy")
        #self.add_class("dataset", 2, "Healthy")

        images_dir = dataset_dir + 'images\\'
        annotations_dir = dataset_dir + 'annots\\'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= 35:
                continue

            if not is_train and int(image_id) < 35:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
		# get details of image
        info = self.image_info[image_id]
		# define box file location
        path = info['annotation']
        #return info, path


		# load XML
        boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]


            # box[4] will have the name of the class
             # Treat all "Dead" instances as "Healthy"
            if box[4] == 'Dead':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Healthy'))
            elif box[4] == 'Healthy':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Healthy'))

              #print('--> ',masks[row_s:row_e, col_s:col_e, i])
            #elif(box[4] == 'orange'):
                #masks[row_s:row_e, col_s:col_e, i] = 3
                #class_ids.append(self.class_names.index('orange'))

        return masks, asarray(class_ids, dtype='int32')

    
    #tree = xml.etree.ElementTree.parse(filename)
    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
		# load and parse the file
        tree = tree = xml.etree.ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text   #Add label name to the box list
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class ShellfishConfig(mrcnn.config.Config):
    NAME = "shellfish_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 131

# Train
train_dataset = ShellfishDataset()
train_dataset.load_dataset(dataset_dir='C:\\Users\\Unyim\\Downloads\\CSDownloads\\Mask-RCNN-TF2\\kangaroo-transfer-learning\\kangaroo\\', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = ShellfishDataset()
validation_dataset.load_dataset(dataset_dir='C:\\Users\\Unyim\\Downloads\\CSDownloads\\Mask-RCNN-TF2\\kangaroo-transfer-learning\\kangaroo\\', is_train=False)
validation_dataset.prepare()

# Model Configuration
shellfish_config = ShellfishConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='.\\', 
                             config=shellfish_config)

model.load_weights(filepath='kangaroo-transfer-learning\\mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])



import imgaug.augmenters as iaa

augmentation = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flips
            iaa.Affine(rotate=(-50, 50)),  # Rotations
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),  # Translations
            iaa.Affine(scale=(1.0, 2.0))  # Scaling
            ], random_order=True) # apply augmenters in random order

print('=========STARTING TRAINGING=========')
model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=shellfish_config.LEARNING_RATE, 
            epochs=3, 
            layers='heads',
            #augmentation=augmentation
            )

model_path = 'shellfish_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
