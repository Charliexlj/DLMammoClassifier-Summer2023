# Week 3: Data Preprocessing, Preliminary Research and Baseline Model Implementation

During the second week of the project we have collected as much as data as we can, and get a getter insight about the project. This week we will focus on data collection, preprocessing, and the initial implementation of the model for baseline trials. This document offers an overview of our progress and outlines the tasks for Week 3.

## Task Division

| Task | Group Member |
| ------------- | ------------- |
| Cloud Operations | Andy |
| Image Preprocessing & Data Exploration | Luca, Lun, and Henry |
| Algorithm Research & Trial Implementation | Charlie and David |

### Cloud Operation

- Set up, manage, and maintain the cloud infrastructure
    - Choose a cloud platform and create a project
    - Enable necessary APIs, set up the relevant cloud services
    - Manage computational resources
- Set up appropriate access controls, monitor usage and cost
- Provide guidlines for other group members on how to use the cloud services
    - File structures
    - Expected procedure to use the virtual machine
    - Data and model access

### Algorithm Research & Trial Implementation
- Research for different algorithms
    - Generative algorithms for data enrichment
        - GANs
            - ViTGAN
        - Diffusion
    - Classification algorithms
        - ViT
        - Localisation
            - ROI
            - Segmentation
        - Local classification
            - VGG16
            - Resnet
    - Pre-train using self-supervised learning
        - Contrasive learning
        - Generation with masked data
- Trial Implementation
    - Complete model architectural design
    - Trial on small dataset with fewer parameters
