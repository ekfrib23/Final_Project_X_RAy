# Thriving Skies:  AI Modeling for X-Ray Baggage Screenings
----
Team Members:  Ryan Busman, Elaine Kwan, Deb Peters, Nick Watkins

<p align="center">
  <img src="Images/Cover_Pres.png" alt="Cover_Image">
</p>

# Table of Contents

Click on any of the titles below to take you to the desired subsection or your choice. 

* [**Overview**](#Overview)  
* [**Project Purpose and Goals**](#Project-Purpose-and-Goals) 
* [**Audience**](#Audience) 
* [**About the Data**](#About-the-Data) 
* [**About the Data Analysis**](#About-the-Data-Analysis) 
* [**Github Directory Structure**](#Github-Directory-Structure)

## Overview
----
### Why Aviation Security Matters
In the wake of the COVID-19 pandemic, a surge in air travel has swept the globe, reigniting the skies with bustling activity. By the close of 2023, air traffic had reclaimed its pre-pandemic vigor, soaring to 94.1% of 2019 levels, with promising projections for the year ahead (IATA, 2023, 2024). The International Air Transport Association (IATA) anticipates a groundbreaking year in 2024, with the global airline passenger industry poised to achieve record growth—encompassing 4.7 billion passengers, 964 billion dollars in revenue, and 49.3 dollars billion in operating profit (IATA, 2023, 2024).  

![IATA_Passenger Load Factor](<Images/Passenger_Load_Factor.png>)
*Figure 1: Passenger load factor (PLF) reflects the utilization of an airline's passenger capacity. As of late 2023, PLF had surged back to nearly pre-pandemic levels across all regions, as revealed by IATA's research.*   


Safeguarding this aviation ecosystem is the mandate of the United States of America's Transportation Security Administration (TSA). Each day, TSA personnel screen approximately 2 million passengers at 440 airports nationwide (TSA, 2022). Every carry-on bag undergoes inspection, totaling 3.3 million screenings daily (TSA, 2024). 

While the scale of TSA operations is undeniably impressive, the agency is not resting on its laurels. Acknowledging the imperatives of efficiency and accuracy, the TSA is pioneering innovative solutions to streamline the security screening process:

> TSA requires detection technologies that effectively and efficiently screen people for concealed explosive threats. Currently, as people move through checkpoints they must remove outerwear, footwear, belts and headwear, slowing the line and decreasing public acceptance. False alarms are frequent, causing inconvenient and intrusive pat-downs and searches. SaS is developing technology that would enable the scanning of walking passengers, acquiring data through most garments and reliably detecting a wider range of prohibited items regardless of concealment. (TSA, 2024, *Screening at Speed*)

False positives not only disrupt the flow of passengers but also pose significant costs and privacy concerns. To mitigate these challenges, precision and swiftness are paramount. Here, the integration of artificial intelligence and machine learning is emerging as a coup, enhancing the accuracy and efficiency of security screening by automatically identifying potential threats (Greenberg & Gehm, 2020).  

<p align="center">
  <img src="Images/APEX_SaS Graphic_ 2018_Approved-1 (1).jpg" alt="Screening_at_Speed">
</p>
*Figure 2: Continuously innovating, the TSA envisions a future where security screening seamlessly integrates advanced technologies like 3-D imaging and X-Ray diffraction to bolster threat detection capabilities.*

### The Invisible Made Visible:  How X-Rays Work
X-rays, like visible light are energy or electromagnetic radiation.  However, unlike traditional visible-spectrum imagery, X-rays can pass through objects.  Therefore, images created from X-rays offer transparent views of solid objects and can reveal hidden things.

<p align="center">
  <img src="Images/EMSpectrum.png" alt="EMSpectrum">
</p>
*Figure 3:  While invisible to the naked eye, X-rays are still part of the electromagnetic spectrum, offering unique insights into the hidden worlds inside and around us.*

Key to X-ray imaging is attenuation—the measure of energy absorption and deflection as X-rays traverse materials. Defined by the linear attenuation coefficient (μ), attenuation varies with material density, with denser substances like metals exhibiting higher absorption levels. Consequently, metallic objects cast darker shadows on X-ray image negative, akin to the silhouettes in a shadow puppet show.

<p align="center">
  <img src="Images/Attenuation.png" alt="Attentuation">
</p>
*Figure 4: Through attenuation, X-ray images reveal the concealed, with dangerous metallic objects emerging as distinct shadows against the backdrop. (Image adapted from Hassan et al., 2020)*

X-ray baggage screening both single-shot and continuous exposure techniques. The latter, prevalent in cabin baggage screening, offers seamless scanning as items glide along conveyor belts, ensuring swift and thorough examination.

###  A Typical Screening Flow in US Airports

<p align="center">
  <img src="Images/Baggage_Flow.png" width="600" height="400">
</p>

*Figure 5: A diagram of a typical security screening flow in a US airport.* 

## Project Purpose and Goals
----
Our thriving skies project charts a course toward enhanced screening procedures, with a keen focus on baggage inspection enhanced with automated object detection algorithms, powered by cutting-edge convolutional neural networks.  

At the project's culmination, a dynamic web interface will emerge, empowering security personnel with real-time assessments of X-ray images, expediting threat identification, and ensuring the skies remain safe for all travelers.

* **Goal 1:  Improve Accuracy**

    Improve accuracy of dangerous object detection in carry-on passenger bags.   

<br/>

* **Goal 2:  A Screening Interface**  

    Create a web-based interactive tool that can be used to assess the likelihood of a dangerous object in baggage

## Audience
----
* **Transportation Security Administration (TSA) Officials**: Personnel involved in airport security operations, policy development, and technology implementation within the TSA.
<br/>
* **Airline Executives and Managers**: Decision-makers within airline companies responsible for ensuring the safety and security of passengers and crew.
<br/>
* **Airport Authorities**: Officials overseeing airport operations, security protocols, and infrastructure development.
<br/>
* **Security Equipment Manufacturers**: Companies involved in the research, design, development, and production of security screening equipment and technologies.
<br/>
* **Aviation Industry Associations**: Organizations representing various stakeholders in the aviation sector, such as the International Air Transport Association (IATA) and Airports Council International (ACI).
<br/>
* **Travelers**: Individuals interested in understanding the efforts being made to enhance aviation security and ensure safe air travel experiences.

# About the Data
----
* __Organization__: Pontificia Universidad Catolica de Chile - Department of Computer Science - The GRIMA Machine Intelligence Group
* __Database Name__:  GDXray+ (or the GRIMA X-ray database)  
* __Data Access__:  https://domingomery.ing.puc.cl/material/gdxray/

![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/a14c8e1d-2a4a-4ce1-931c-b3a14d9fdc0d)

* __Data Characteristics__: 
    - 21,100 X-ray images organized into 5 groups.  Each group has several series.  
    - Images are saved as Portable Network Graphics (png) 8 -bit grayscale format. 
    - "Additional metadata for each series (such as description of the objects, parameters and description of X-ray imaging system, etc.) are given in an ASCII file called Xssss readme.txt included in sub-folder Xssss, e.g., C0003 readme.txt for series Castings/C0003." (Mery et al., 2015, pg. 4) 

<p align="center">
  <img src="Images/Subsets.png" alt="Subsets_with_series">
</p> 

*Figure 6: The GDXray+ five groups of data with their series. (Image adapted from Mery et al., 2015)*   

* __Data Subset__:  Baggage (3.048GB)


> **Licensure and Credits:**
> -  The X-ray images included in GDXray+ can be used free of charge, for research and educational purposes only. 
> - Redistribution and commercial use is prohibited. Any researcher reporting results which use this database should acknowledge the GDXray+ database by citing:
>   >Mery, D.; Riffo, V.; Zscherpel, U.; Mondragón, G.; Lillo, I.; Zuccar, I.; Lobel, H.; Carrasco, M. (2015): GDXray: The database of X-ray images for nondestructive testing. Journal of Nondestructive Evaluation, 34.4:1-12. [ PDF ]**

# About the Data Analysis
----
## Data Fetching and Preprocessing
* Gather a large dataset of  X-ray images containing both benign and prohibited items (i.e., weapons, explosives, and sharp objects).
* Annotate the images to label benign objects and the specific type of prohibited object.  
* Clean and preprocess the X-ray  images to enhance their quality.
* Normalize pixel values and  resize images to a consistent resolution.

*Comment out any of the below installations if already installed.*
<pre><code>
# pip install tensorflow
# pip install keras
# pip install opencv-python 
</code></pre>

*Import dependencies*
*Load and inspect images using the Python Imaging Library (PIL)*
<pre><code>
from PIL import Image
import os
import numpy as np
import cv2
import pickle
</code></pre>

*Preprocess Images for Machine Learning.*  
*Resize the images to a consistent size.* 
*Specify the target size.*
<pre><code>
target_size = (224, 224)

# Initialize an empty list to store resized image arrays
resized_image_arrays = []

# Iterate over the image arrays
for image_array in image_arrays:
    # Convert the image array back to PIL image
    image = Image.fromarray(image_array)
    
    # Resize the image
    #resized_image = image.resize(target_size)
    resized_image = image.resize(target_size, resample = Image.LANCZOS)
    # Convert the resized image back to NumPy array
    resized_image_array = np.array(resized_image)#.astype(np.float32)
    # append image to resized_image_arrays
    resized_image_arrays.append(resized_image_array)
    
# Print the resized image array
print(resized_image_arrays)
</code></pre>

*Remove unwanted noise in the images to smooth them out (i.e., less grainy).*
*When looking at denoising techniques, we settled on the OpenCV library.*
*Normalize data.*  
*Create an empty list to store the denoised and normalized image arrays.*
<pre><code>
denoised_image_arrays = []
#counter
i = 0

# Iterate over the image arrays
for image_array in resized_image_arrays:
    # print sets of 50 images
    if i%50==0:
        print(f"reading image number {i}")
    # Apply denoising filter and normalizing
    denoised_image_array = cv2.fastNlMeansDenoising(image_array, None, h=10, templateWindowSize=7, 
                                                searchWindowSize=21).astype(np.float32)/255
    
    denoised_image_array = denoised_image_array.reshape(image_array.shape)
    # print denoised_image_arraye
    #display(denoised_image_array)
    # Append the denoised image array to the list
    denoised_image_arrays.append(denoised_image_array)

    # add 1 to counter
    i=i+1
# Print the denoised image arrays
for i, denoised_image_array in enumerate(denoised_image_arrays):
    print(f"Values of denoised image {i}:", denoised_image_array)
</code></pre>

## Data Augmentation 
* Augment the dataset by  applying transformations (rotation, scaling, etc.) to create additional  training samples. 

*Import dependencies*
*Load and inspect images using the Python Imaging Library (PIL)*
<pre><code>
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
</code></pre>

*Augmentation function*
<pre><code>
def augment(img):
    #call function to reshape the image
    reshaped_image_array = reshape(img)
        
    # Create the ImageDataGenerator object with desired augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Generate augmented images
    augmented_images = []
    
    for _ in range(5):  # Augment the image 5 times for variety
        augmented_image = next(datagen.flow(reshaped_image_array, batch_size=1))[0]
        augmented_images.append(augmented_image)

    # uncomment to show images
    # Visualize the original and augmented images
#     plt.figure(figsize=(12, 6))
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)
#         if i == 0:
#             plt.imshow((reshaped_image_array[0, :, :, 0]*255).astype('uint8'), cmap='gray')  # Original image
#         else:
#             plt.imshow((augmented_images[i - 1][:, :, 0]*255).astype('uint8'), cmap='gray')
#         plt.axis('off')

    #display images
   # plt.show()
    # return reshaped images
    return augmented_images
</code></pre>


*Create an empty list for both X and y label augmentations*
<pre><code>
X_train_aug = []
y_train_aug = []
# Loop through each image in the training data
for i in range(len(X_train)):
    # Select the image and its y label
    imgX = X_train[i]
    label = y_train[i]
    
    # Add a channel dimension for grayscale images
    imgX = augment(imgX)
    
    # Use a loop to create 5 new images and append augmented images 
    X_train_aug.extend(imgX) 
    for j in range(5):
        y_train_aug.append(label)

# Print the lengths of both augmented sets to ensure they are the same length
print(f'length of X training data {len(X_train_aug)} and y training data {len(y_train_aug)}')
</code></pre>

## Model Selection and Architecture
* Choose an appropriate deep  learning model for object detection. 
* Design a custom model or work  with a pre-trained model.
* Add layers for object  detection and classification.

*Define input shape*
*Define CNN model*
<pre><code>
input_shape = X_test_np[0].shape
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='sigmoid')  # 5 classes of images
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32#64
epochs = 6
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs
)
</code></pre>

## Training and Testing
* Split the dataset into  training, validation, and test sets.
* Train the model using the  annotated X-ray images.
* Optimize hyperparameters  (learning rate, batch size, etc.).


****NOTE TO RYAN:  PUT TRAINTEST.png under Images subdirectory here, instead.****





## Evaluation and Fine Tuning

* Evaluate the model’s  performance on the validation and test sets.
* Metrics could include  precision, recall, F1-score, and accuracy.  Test for bias.
* Fine-tune the model based on evaluation results.
* Address false positives and false negatives, as needed.

****NOTE TO RYAN:  PUT Loss.png under Images subdirectory here, instead.****

![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/7d387c06-5ad6-47fa-a007-5c9c30c0ae12)


****NOTE TO RYAN:  PUT Accuracy.png under Images subdirectory here.****


****NOTE TO RYAN:  PUT ConfusionMatrix.png under Images subdirectory here, instead.****
![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/2c8174cb-33e8-442c-b853-b64330ae9592)


## Model Implementation

The Jupyter notebooks will need to be opened and ran in the following order:

1. Data_Cleaning_Preprocessing
2. Data_Augmentation
3. Modeling
4. pickle2png
5. Implementation

This is how a user can test a prediction using an image from "Images_for_Testing" subdiretory in the Github repository:  

1. The user chooses from one of 678 test images selected from the data set.
2. The user chooses an image number.
3.  The model predicts if the image is benign (harmless), a gun, a knife, a razorblade or a shuriken (throwing star.)

****NOTE TO RYAN:  PUT Prediction.png under Images subdirectory here, instead.****
![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/5f80a050-99fb-4f2e-a8d5-ced69d7fcd79)

![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/73197d66-3254-40f2-ad36-9117f5b7e61c)

![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/016e36b7-7dd8-425c-9713-9fd6ec8b6d48)

![image](https://github.com/Rbusman/Thriving_Skies/assets/146746454/628d0b31-2148-4186-8216-31ff68fd5d68)





## Github Directory Structure
---
* **Code**:  contains all code by topic
* **Data**:  contains all datasets by topic
* **Images**: contains images for the final presentation and README.md
* **Presentation**: contains the final presentation 
* **Sources**:  secondary research literature
* **Proposal**:  the initial proposal for the project
* **README.md**:  contains project details and definitions


