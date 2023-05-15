# Week 3: Data Preprocessing, Preliminary Research and Baseline Model Implementation

During the second week of the project we have collected as much as data as we can, and get a getter insight about the project. This week we will focus on data collection, preprocessing, and the initial implementation of the model for baseline trials. This document offers an overview of our progress and outlines the tasks for Week 3.

## Task Division

| Task | Group Member |
| ------------- | ------------- |
| Cloud Operations | Andy |
| Image Preprocessing & Data Exploration | Luca, Lun, and Henry |
| Algorithm Research & Preliminary Implementation | Charlie and David |

### Cloud Operation

- Establish, administer, and sustain the cloud infrastructure:
    - Select a suitable cloud platform and initiate a project.
    - Activate the necessary APIs and configure relevant cloud services.
    - Manage computational resources.
- Set up appropriate access controls, monitor usage and costs.
- Provide guidlines for team members about utilizing the cloud services:
    - Establish file structures.
    - Specify the standard procedure for utilizing the virtual machine.
    - Arrange access to data and models.

### Algorithm Research & Preliminary Implementation
- Conduct research on various algorithms:
    - Generative algorithms for data enrichment
        - GANs: ViTGAN
        - Diffusion
    - Classification algorithms
        - ViT
        - Localisation: ROI, Segmentation(U-Net)
        - Local classification: VGG19, Resnet
    - Pre-train models using self-supervised learning approaches
        - Contrastive learning
        - Generation using masked data
- Preliminary Implementation
    - Design the complete model architecture
    - Conduct preliminary tests on a smaller dataset with fewer parameters
