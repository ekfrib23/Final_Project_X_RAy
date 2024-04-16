# AI Modeling for X-Ray Baggage Screenings
----
Team Members:  Ryan Busman, Elaine Kwan, Deb Peters, Nick Watkins

![Placeholder - Presentation_Cover_Image](<Presentation/Images/insert_file_name_here.png>)

# Overview
----
### The Significance of the Aviation Security
In 2019, the aviation industry generation 65.5 million jobs. Its share in the global economy was estimated at $ 2.7 trillion or 3.6% of the world's gross domestic product. In the same year, 4.3 billion passengers were carried by airlines worldwide, which was an increase of 6.4% from 2017. Just in the US, there were over one billion air passengers in 2019: an increase of 3.8 % compared with 2018 (Statistica, 2019).

![Placeholder - Graph or Stats ](<Presentation/Images/insert_file_name_here.png>)

NOTE:  More here on present state in the US and security's relationship to costs.... Though other tech proposed... X-Ray screening is still the go-to because (advantages)....?  Disadvantages of X-Ray... What about the rate of fasle positives?  Is that expensive and why?  AI paired with shows promise... 

### How X-Rays Work
Unlike traditional visible-spectrum images (i.e., a photograph or selfie), X-ray images offer transparent views of objects.  X-ray attenuation and image intensity influenced by various factors. The attenuation of X-rays as they pass through objects is determined by the linear attenuation coefficient (μ), with denser materials resulting in higher attenuation levels. Consequently, materials such as metals attenuate X-rays more effectively, resulting in darker, less intensified images. 



![Placeholder - Attenuation and Intensity](<Presentation/Images/insert_file_name_here.png>)

Radiographic X-ray imaging serves as the primary method for baggage screening, utilizing either single radiographic shots or continuous exposure with line sensors. The latter approach, commonly employed for cabin baggage screening travelling continuously on conveyer belts, facilitates seamless scanning and storage of X-ray images 



NOTE:  So what?  CHALLENGES OF OVERLAPPING OBJECTS?  LESS INTENSITY LEADS TO MORE FALSE POSITIVES?  LESS EFFICIENCEY


![Placeholder - Radiographic XRay Types](<Presentation/Images/insert_file_name_here.png>)


###  Aviation Security Policies

NOTE:  We need to have this source and correct the below to reflect that source.  We can then cite and reference the source

Common basic Standard comprise (US Govt, Accountability Office, 2004)

* screening of passengers, cabin baggage and hold (checked) baggage 
* airport security (access control, surveillance) 
* aircraft security checks and searches 
* screening of cargo and mail — screening of airport supplies 
* staff recruitment and training. 

There has been a need for the screening of passengers and their baggage for three main purposes: 

* the illegal movement of goods or prohibited items according to the local legislative procedures 
*  fraud and revenue avoidance 
* terrorist threat

###  Typical Screening Flow in US Airports

Efficiency and accuracy are fundamental principles guiding airport security operations. To achieve these objectives, airports across the United States employ state-of-the-art X-ray-based screening technologies. Both 2D and 3D X-ray scanners are deployed for screening both hold and cabin baggage, ensuring comprehensive threat detection. 

While X-ray diffraction (XRD) technology may be utilized in certain cases, its current deployment remains limited (US Government Accountability Office, Year). 

NOTE:  How so?   To the public, it seems like X-ray scanning of baggage is everywhwere?  


Given the dynamic nature of security threats, continuous innovation is essential to adapt screening procedures to evolving circumstances.

![Flow](<Presentation/Images/flow.png>)

# Project Purpose and Goals
----
This project aims to enhance security screening procedures through the utilization of advanced X-ray technology, with a specific focus on baggage inspection. While existing screening methods such as walk-through metal detectors and full-body scanners are effective for passenger screening, the primary objective here is to detect and mitigate potential threats concealed within baggage. This endeavor will involve a combination of visual inspection techniques and automated object detection algorithms.

... Used a subset of what database?...

*Goal 1*:  Improve accuracy .... bla bla bla... 
*Goal 2*:  Interactive tool... bla bla bla... 

....Implications or aviation security bla bla bla... 

# Audience
----
...bla bla bla... Who cares now and why?

# About the Data
----
* __Organization__: ...
* __Data Access__:  ... website..
* __Data Type__:
* __Data Subsets__:
* __Data Characteristics__:

![Placeholder - Sample images from baggage subset](<Presentation/Images/insert_file_name_here.png>)


...Quotes on origin of database, validity, and reliability/used how often of dataset

...which subset used for this study... how was the data sampled...? 

> **Licensure and Credits:**
> - ...

 *References*
    * Statistica. (2019). *Aviation industry - statistics & facts.* Retrieved from https://www.statista.com/topics/1707/air-transportation/#topicOverview
    * US Government Accountability Office. (2004). *Aviation security: Challenges exist in stabilizing and enhancing passenger and baggage screening operations.* Retrieved from https://www.gao.gov/products/gao-04-440t

# About the Data Analysis
----
## Data Collection
* Gather a large dataset of  X-ray images containing both normal and prohibited items (such as  weapons, explosives, and sharp objects).
* Annotate the images to label  the presence or absence of prohibited items.


## Preprocessing
* Clean and preprocess the X-ray  images to enhance their quality.
* Normalize pixel values and  resize images to a consistent resolution.

 

## Model Selection 
* Choose an appropriate deep  learning model for object detection. 


## Model Architecture 
* Design a custom model or work  with a pre-trained model.
* Add layers for object  detection and classification.

## Data Augmentation 
* Augment the dataset by  applying transformations (rotation, scaling, etc.) to create additional  training samples

## Training 
* Split the dataset into  training, validation, and test sets.
* Train the model using the  annotated X-ray images.
* Optimize hyperparameters  (learning rate, batch size, etc.).

## Evaluation

* Evaluate the model’s  performance on the validation and test sets.
* Metrics could include  precision, recall, F1-score, and accuracy.  Test for bias.  

## Fine-Tuning:

* Fine-tune the model based on  evaluation results.
* Address false positives and  false negatives.

## Model Implementation

# Directory Structure
---
- ... 

