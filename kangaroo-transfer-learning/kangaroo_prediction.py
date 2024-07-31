import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG','Healthy']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="shellfish_mask_rcnn_trained.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("kangaroo-transfer-learning\\kangaroo\\Ground_Truth.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]
arr  = r['masks']
np.save('GroundTruthMasks', arr)# save our masks to later be plotted

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])

#np.save('MYYYDATTTTA.npy', r['masks'])
# classification pipeline
'''import cv2 as cv
import numpy as np

# Get the number of detected instances
num_instances = r['masks'].shape[2]

for i in range(num_instances):
    # Extract the individual mask
    individual_mask = r['masks'][:, :, i]
    
    # Create a copy of the original image
    masked_image = image.copy()
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    
    # Apply the mask to the image
    # Here, we create a red color for the mask
    red_mask = np.zeros_like(masked_image)
    red_mask[individual_mask] = [255, 0, 0]  # Red color
    

    masked_image = make_detections(individual_mask,masked_image,model)
    # Combine the red mask with the original image
    #masked_image = cv2.addWeighted(masked_image, 1, red_mask, 0.5, 0)
    
    # Display the mask and the masked image using OpenCV
    #cv2.imshow(f"Mask {i + 1}", individual_mask.astype(np.uint8) * 255)  # Show mask
    #cv2.imshow(f"Masked Image {i + 1}", masked_image)  # Show masked image
    
    #cv2.waitKey(0)  # Wait for a key press to proceed to the next mask
    #cv2.destroyAllWindows()  # Close all OpenCV windows


def make_detections(mask,img,model):
    #circle_mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    #poly_mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

    #for circle in circles[0, :]:
        #circle_mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
        #x = circle[0]
        #y = circle[1]
        #rad = circle[2]

        #mask = cv.circle(np.zeros(img.shape[:2], dtype=np.uint8), (x,y), rad, (255, 255, 255), -1)
        
        # Apply the mask to the image
        roi = cv.bitwise_and(img, img, mask=mask)
        mean_bgr = cv.mean(roi, mask=mask)[:3]
        hsv_image = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mean_hsv = cv.mean(hsv_image, mask=mask)[:3]


        
        features = [[mean_bgr[0],mean_bgr[1],mean_bgr[2],mean_hsv[0],mean_hsv[1],mean_hsv[2]]]
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            img = cv.putText(img, 'Dead', (x+rad,y+rad), cv.FONT_HERSHEY_SIMPLEX ,  1, (0, 0, 255), 1, cv.LINE_AA)
            cv.circle(img, (x, y), rad, (0, 0, 255), 2)
        else:
            img = cv.putText(img, 'Alive', (x+rad,y+rad), cv.FONT_HERSHEY_SIMPLEX ,  1, (255, 0, 0), 1, cv.LINE_AA)
            cv.circle(img, (x, y), rad, (255, 0, 0), 2)
        print(prediction)'''

# classification pipeline

#np.save('masks.npy', r['masks'])