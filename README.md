# DLMammoClassifier-Summer2023
Imperial College London Summer 2023 Group Consultancy Project

## Overview

This repository contains the code and resources for the development of a mammography classification system using deep learning techniques. The project is carried out by a group of students from Imperial College London during the Summer of 2023.

## Timeline and Work Distribution

- [Week 1: Client consultation and project planning](docs/week1.md)
- [Week 2: Data collection and exploration](docs/week2.md)

## Week 1: Client Consultation

### Questions for Clients

1. What software and hardware resources are available for this project?
2. Is there a budget allocated for additional resources (e.g., cloud computing services, data storage)?
3. What are the expected deliverables and performance metrics for the final model?
4. Are there any specific data formats, platforms, or tools that the clients prefer to work with?
5. What is the desired level of interpretability for the model?
6. Are there any privacy or security requirements for handling the data?

(Add more questions as needed.)

## Model Building Pipeline

1. Collect data: Gather mammography images from public datasets and/or client-provided sources.
2. Preprocessing:
   - Augmentation: Apply data augmentation techniques to increase the size and diversity of the dataset.
   - Resize: Resize images to a consistent resolution for input to the model.
3. Pretrain with self-supervised learning: Train a base model using self-supervised learning techniques to learn useful image representations.
4. Fine-tune the model: Fine-tune the pretrained model using labeled data to optimize it for mammography classification.
5. Evaluate the model: Assess the performance of the model using appropriate metrics (e.g., accuracy, F1 score, sensitivity, specificity).
6. Optimize and deploy: Optimize the model for deployment, and integrate it into a suitable platform for the clients.