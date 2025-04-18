#!/usr/bin/env python
# coding: utf-8

# # import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 
# # Set paths for training and validation datasets
# train_dir = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/train'
# val_dir = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/val'
# 
# # Preprocessing for training and validation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,        # Normalize pixel values
#     rotation_range=20,     # Augmentation: rotate images
#     width_shift_range=0.2, # Augmentation: shift width
#     height_shift_range=0.2, # Augmentation: shift height
#     horizontal_flip=True   # Augmentation: flip horizontally
# )
# 
# val_datagen = ImageDataGenerator(rescale=1./255)  # Validation data doesn't need augmentation
# 
# # Load datasets
# train_data = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(128, 128),  # Resize all images to 128x128
#     batch_size=32,
#     class_mode='binary'      # Binary classification: breed vs. stray
# )
# 
# val_data = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary'
# )
# 

# In[14]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/train'
val_dir = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/val'
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    rotation_range=20,     # Augmentation: rotate images
    width_shift_range=0.2, # Augmentation: shift width
    height_shift_range=0.2, # Augmentation: shift height
    horizontal_flip=True 
)# Augmentation: flip horizontally
val_datagen = ImageDataGenerator(rescale=1./255)  # Validation data doesn't need augmentation
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize all images to 128x128
    batch_size=32,
    class_mode='binary'      # Binary classification: breed vs. stray
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)


# In[10]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load MobileNetV2 base model with pretrained weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
base_model.trainable = False

# Build the classification model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (breed vs. stray)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()


# In[7]:


import tensorflow as tf
print(tf.__version__)


# In[15]:


# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Save the trained model
model.save('breed_vs_stray_model.h5')


# In[16]:


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[7]:


import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Function to classify and annotate the image
def classify_and_annotate_image(img_path, output_path):
    # Load the image
    img = cv2.imread(img_path)
    original_img = img.copy()  # Keep a copy for annotation

    # Resize the image for the model
    resized_img = cv2.resize(img, (128, 128))
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

    # Annotate the image
    height, width, _ = img.shape
    text = f"Prediction: {label}"
    font_scale = 0.8
    font_thickness = 2
    text_color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray

    # Draw a rectangle to annotate the detected dog
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), text_color, 2)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Image saved with annotation: {output_path}")

    # Display the annotated image using Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
    plt.imshow(img_rgb)
    plt.axis('off')  # Turn off the axis
    plt.title(text)  # Add the prediction as the title
    plt.show()

# Example usage
classify_and_annotate_image(
    '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/val/stray/dogs_100_jpg.rf.815826bca3ce45d3184ad28f91e25276.jpg',
    '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/annotated_image.jpg'
)


# In[11]:


import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Function to classify and annotate the image
def classify_and_annotate_image(img_path, output_path):
    # Load the image
    img = cv2.imread(img_path)
    original_img = img.copy()  # Keep a copy for annotation

    # Resize the image for the model
    resized_img = cv2.resize(img, (128, 128))
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

    # Annotate the image
    height, width, _ = img.shape
    text = f"Prediction: {label}"
    font_scale = 0.8
    font_thickness = 2
    text_color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray

    # Draw a rectangle to annotate the detected dog
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), text_color, 2)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Image saved with annotation: {output_path}")

    # Display the annotated image using Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
    plt.imshow(img_rgb)
    plt.axis('off')  # Turn off the axis
    plt.title(text)  # Add the prediction as the title
    plt.show()

# Example usage
classify_and_annotate_image(
    '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/train/pet/n02093428_814.jpg',
    '/Users/Guru Kiran/OneDrive/Desktop/jupyter/dataset/annotated_image.jpg'
)


# In[3]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO  # Import YOLOv8
import matplotlib.pyplot as plt
from IPython.display import Video

# Load the trained model (Breed vs Stray classification)
classification_model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Load YOLOv8 model for dog detection
yolo_model = YOLO('yolo11n.pt')  # Replace with your YOLO model path or variant

# Function to preprocess the cropped dog image for classification
def preprocess_frame(cropped_img):
    resized_img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input size
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to process video and classify dogs
def classify_and_annotate_video(input_video_path, output_video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    # Check if video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or video ended.")
            break

        # Use YOLO to detect objects in the frame
        results = yolo_model(frame)

        # Loop through detections to find dogs (class 16 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # Class ID
                if cls == 16:  # Class 16 corresponds to 'dog'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the detected dog's region for classification
                    dog_crop = frame[y1:y2, x1:x2]

                    try:
                        # Preprocess cropped image for classification
                        img_array = preprocess_frame(dog_crop)

                        # Classify the cropped image
                        prediction = classification_model.predict(img_array)
                        label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

                        # Annotate the frame
                        color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

        # Write annotated frame to output video
        out.write(frame)

        # Display frame in the notebook (optional, for debugging)
        # Convert BGR to RGB for displaying in Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing completed. Output saved to {output_video_path}")

# Example usage
input_video_path = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/4300700-hd_1920_1080_30fps.mp4'  # Replace with your input video path
output_video_path = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/output_video.avi'  # Replace with your desired output path
classify_and_annotate_video(input_video_path, output_video_path)

# Display the processed video inline
Video(output_video_path, embed=True)


# In[12]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO  # Import YOLOv8
import matplotlib.pyplot as plt
from IPython.display import Video

# Load the trained model (Breed vs Stray classification)
classification_model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Load YOLOv8 model for dog detection
yolo_model = YOLO('yolo11n.pt')  # Replace with your YOLO model path or variant

# Function to preprocess the cropped dog image for classification
def preprocess_frame(cropped_img):
    resized_img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input size
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to process video and classify dogs
def classify_and_annotate_video(input_video_path, output_video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    # Check if video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or video ended.")
            break

        # Use YOLO to detect objects in the frame
        results = yolo_model(frame)

        # Loop through detections to find dogs (class 16 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # Class ID
                if cls == 16:  # Class 16 corresponds to 'dog'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the detected dog's region for classification
                    dog_crop = frame[y1:y2, x1:x2]

                    try:
                        # Preprocess cropped image for classification
                        img_array = preprocess_frame(dog_crop)

                        # Classify the cropped image
                        prediction = classification_model.predict(img_array)
                        label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

                        # Annotate the frame
                        color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

        # Write annotated frame to output video
        out.write(frame)

        # Display frame in the notebook (optional, for debugging)
        # Convert BGR to RGB for displaying in Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing completed. Output saved to {output_video_path}")

# Example usage
input_video_path = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/854383-hd_1920_1080_30fps.mp4'  # Replace with your input video path
output_video_path = '/Users/Guru Kiran/OneDrive/Desktop/jupyter/output_video1.avi'  # Replace with your desired output path
classify_and_annotate_video(input_video_path, output_video_path)

# Display the processed video inline
Video(output_video_path, embed=True)


# In[14]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO  # Import YOLOv8

# Load the trained model (Breed vs Stray classification)
classification_model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Load YOLOv8 model for dog detection
yolo_model = YOLO('yolo11n.pt')  # Replace with your YOLO model path or variant

# Function to preprocess the cropped dog image for classification
def preprocess_frame(cropped_img):
    resized_img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input size
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to process live video and classify dogs
def classify_and_annotate_live_video():
    # Open webcam or live video feed (use 0 for the default webcam, or change to the index of your camera)
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or a video stream URL

    # Check if video capture is opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera or video feed.")
        return

    print("Press 'q' to exit the live video.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, skipping...")
            continue

        # Use YOLO to detect objects in the frame
        results = yolo_model(frame)

        # Loop through detections to find dogs (class 16 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # Class ID
                if cls == 16:  # Class 16 corresponds to 'dog'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the detected dog's region for classification
                    dog_crop = frame[y1:y2, x1:x2]

                    try:
                        # Preprocess cropped image for classification
                        img_array = preprocess_frame(dog_crop)

                        # Classify the cropped image
                        prediction = classification_model.predict(img_array)
                        label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

                        # Annotate the frame
                        color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

        # Display the annotated frame
        cv2.imshow('Live Video Feed', frame)

        # Press 'q' to quit the live video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start live video detection
classify_and_annotate_live_video()


# In[1]:


get_ipython().system('pip install twilio')


# In[2]:


#Live Feed 
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO  # Import YOLOv11
from twilio.rest import Client  # Import Twilio
import time  # For cooldown timer

# Twilio Configuration
TWILIO_ACCOUNT_SID = ''  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = ''  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = ''  # Your Twilio phone number
RECIPIENT_PHONE_NUMBER = ''  # Recipient's phone number

# Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Cooldown for SMS notifications
last_notification_time = 0  # Timestamp of the last notification
notification_cooldown = 60  # Cooldown period in seconds

# Load the trained model (Breed vs Stray classification)
classification_model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Load YOLOv8 model for dog detection
yolo_model = YOLO('yolo11n.pt')  # Replace with your YOLO model path or variant

# Function to preprocess the cropped dog image for classification
def preprocess_frame(cropped_img):
    resized_img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input size
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to send SMS notifications using Twilio
def send_sms_notification(message):
    global last_notification_time  # Access the global variable
    current_time = time.time()  # Get the current time

    # Check if the cooldown period has passed
    if current_time - last_notification_time >= notification_cooldown:
        try:
            message = client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                body=message,
                to=RECIPIENT_PHONE_NUMBER
            )
            print(f"Notification sent successfully! SID: {message.sid}")
            last_notification_time = current_time  # Update the last notification time
        except Exception as e:
            print(f"Failed to send notification: {e}")
    else:
        remaining_time = int(notification_cooldown - (current_time - last_notification_time))
        print(f"Notification skipped. Please wait {remaining_time} seconds before sending another.")

# Function to process live video and classify dogs
def classify_and_annotate_live_video():
    # Open webcam or live video feed (use 0 for the default webcam)
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or a video stream URL

    # Check if video capture is opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera or video feed.")
        return

    print("Press 'q' to exit the live video.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, skipping...")
            continue

        # Use YOLO to detect objects in the frame
        results = yolo_model(frame)

        # Loop through detections to find dogs (class 16 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # Class ID
                if cls == 16:  # Class 16 corresponds to 'dog'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the detected dog's region for classification
                    dog_crop = frame[y1:y2, x1:x2]

                    try:
                        # Preprocess cropped image for classification
                        img_array = preprocess_frame(dog_crop)

                        # Classify the cropped image
                        prediction = classification_model.predict(img_array)
                        label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

                        # Annotate the frame
                        color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label

                        # Send notification if a stray dog is detected
                        if label == 'Stray Dog':
                            send_sms_notification("Stray Dog detected in the live video feed!")

                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

        # Display the annotated frame
        cv2.imshow('Live Video Feed', frame)

        # Press 'q' to quit the live video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start live video detection
classify_and_annotate_live_video()


# In[8]:


import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO  # Import YOLOv11
from twilio.rest import Client  # Import Twilio
import time  # For cooldown timer

# Twilio Configuration
TWILIO_ACCOUNT_SID = ''  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = ''  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = ''  # Your Twilio phone number
RECIPIENT_PHONE_NUMBER = ''  # Recipient's phone number

# Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Cooldown for SMS notifications
last_notification_time = 0  # Timestamp of the last notification
notification_cooldown = 60  # Cooldown period in seconds

# Load the trained model (Breed vs Stray classification)
classification_model = tf.keras.models.load_model('breed_vs_stray_model.h5')

# Load YOLOv8 model for dog detection
yolo_model = YOLO('yolo11n.pt')  # Replace with your YOLO model path or variant

# Function to preprocess the cropped dog image for classification
def preprocess_frame(cropped_img):
    resized_img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input size
    img_array = resized_img / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to send SMS notifications using Twilio
def send_sms_notification(message):
    global last_notification_time  # Access the global variable
    current_time = time.time()  # Get the current time

    # Check if the cooldown period has passed
    if current_time - last_notification_time >= notification_cooldown:
        try:
            message = client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                body=message,
                to=RECIPIENT_PHONE_NUMBER
            )
            print(f"Notification sent successfully! SID: {message.sid}")
            last_notification_time = current_time  # Update the last notification time
        except Exception as e:
            print(f"Failed to send notification: {e}")
    else:
        remaining_time = int(notification_cooldown - (current_time - last_notification_time))
        print(f"Notification skipped. Please wait {remaining_time} seconds before sending another.")

# Function to process live video and classify dogs
def classify_and_annotate_live_video():
    # Open webcam or live video feed (use 0 for the default webcam)
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or a video stream URL

    # Check if video capture is opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera or video feed.")
        return

    print("Press 'q' to exit the live video.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, skipping...")
            continue

        # Use YOLO to detect objects in the frame
        results = yolo_model(frame)

        # Loop through detections to find dogs (class 16 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # Class ID
                if cls == 16:  # Class 16 corresponds to 'dog'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the detected dog's region for classification
                    dog_crop = frame[y1:y2, x1:x2]

                    try:
                        # Preprocess cropped image for classification
                        img_array = preprocess_frame(dog_crop)

                        # Classify the cropped image
                        prediction = classification_model.predict(img_array)
                        label = 'Breed Dog' if prediction[0][0] < 0.5 else 'Stray Dog'

                        # Annotate the frame
                        color = (0, 255, 0) if label == 'Breed Dog' else (0, 0, 255)  # Green for breed, red for stray
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label

                        # Send notification if a stray dog is detected
                        if label == 'Stray Dog':
                            send_sms_notification("Stray Dog detected in the live video feed!")

                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

        # Display the annotated frame
        cv2.imshow('Live Video Feed', frame)

        # Press 'q' to quit the live video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start live video detection
classify_and_annotate_live_video()


# In[ ]:




