# Week 7: ResNet Pretraining, Fine-tuning, and Snip Dataset Generation

During the sixth week of the project, we made progress in fine-tuning the network, implementing evaluation code, visualizing the data, and establishing a project setup procedure. In the seventh week, we began working on parallel tasks to enhance our model's capabilities. This included building a ResNet for pretraining, fine-tuning the network further, and creating a dataset that extracts patches labelled dataset for training and testing. This document provides an overview of our progress and outlines the tasks accomplished during Week 7.

### 1. ResNet Pretraining:
   - We initiated the development of a ResNet architecture for pretraining. Pretraining the ResNet on a large-scale patches dataset allows us to capture high-level features and transfer this knowledge to our detection task.

### 2. Fine-tuning:
   - We continued fine-tuning the network to further improve its performance on the detection task.
   - We utilized our labelled dataset and snipped out patches to create our own dataset for fine-tuning.
   ```
   if self.stage == 'local':
        filename = ...
        roi = ...
        patch, no_flag = process_images_patch(image, roi, size=56)
        if 'Benign' in filename:
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        return patch, label.float()
   ```
   ![Patch Extration](./res/patch.png)

## Work Summary
By initiating ResNet pretraining, performing further fine-tuning, and generating a snip dataset from the labelled dataset, we are taking parallel steps to improve the model's capabilities. The ResNet pretraining enhances the model's feature extraction abilities, while the fine-tuning process adapts it to our specific detection task. Additionally, the snip dataset provides additional training and testing samples, potentially improving the model's accuracy and robustness. In the upcoming weeks, we will continue optimizing the model, evaluating its performance, and preparing for the final stages of the project.