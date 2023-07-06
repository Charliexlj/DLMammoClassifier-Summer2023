# Week 6: Autoencoder Training, Network Fine-tuning, Evaluation Code, Data Visualization, and Project Setup Procedure

During the fifth week of the project, we made progress in training the encoder, testing loss functions, adjusting hyperparameters, and developing the network components. We also finalized the project file structure based on our growing insights and familiarity with the platform. In the sixth week, our focus shifted towards fine-tuning the network, implementing evaluation code for each stage of the model, visualizing the data, and establishing a comprehensive project setup procedure. This document provides an overview of our progress and outlines the tasks accomplished during Week 6.

1. Autoencoder Training:
   - We started the process of pretraining the UNet as an autoencoder using mask-reconstruction. Thanks for the previous work on modularisation of the code, the implementation is relatively simple as it can use the same structure as training the encoder part.

2. Network Fine-tuning:
   - We started the process of fine-tuning the network to improve its performance on our specific detection task. Fine-tuning involves training the entire network using our task-specific dataset.
   - We finalise on using IOU loss instead of Cross Entropy Loss after extensive research since:
        - Our goal is to find the region but not a refined boarder, we use this location to the rest of the network for further classification.
        - IOU loss will produce a more rounded segmentation, focus on area but not specific pixels, this would be ideal for finding the location of the region.
        ```
        # Our custom soft IOU loss
        class mIoULoss(nn.Module):
            def __init__(self, weight=None, size_average=True, n_classes=2):
                super(mIoULoss, self).__init__()
                self.classes = n_classes

            def forward(self, inputs, target_oneHot):
                # inputs => N x Classes x H x W
                # target_oneHot => N x Classes x H x W
                N = inputs.size()[0]
                inputs = nn.functional.softmax(inputs, dim=1)
                inter = inputs * target_oneHot
                inter = inter.view(N, self.classes, -1).sum(2)
                union = inputs + target_oneHot - (inputs*target_oneHot)
                union = union.view(N, self.classes, -1).sum(2)
                loss = inter/union
                return 1-loss.mean()
        ```

3. Evaluation Code:
   - To assess the performance of each stage of the model, we implemented evaluation code for them. This code allows us to compute relevant metrics and corresponding data visuallisation at different stages of the detection pipeline.

4. Data Visualization:
   - We integrated data visualization techniques into our workflow to gain better insights into the dataset and the model's predictions. 
   - We use pyplot to visualize the logits and the targets to inspect the differences at different stage of training, this is a more direct evaluation compare to losses.

5. Project Setup Procedure:
   - To ensure that our project can be easily launched on any new machine or environment, we established a comprehensive project setup procedure. This procedure includes a clear set of instructions for setting up the required dependencies, libraries, and configurations. By documenting the setup process, we ensure that the project can be replicated and deployed without obstacles.
        - We add setup.py parallel to Mammolibs for easy installation of the custom library
        - We include a requirements.txt for all the libraries this project depends on, including Mammolibs
        - We made every dependency of file root on porject current directory, so regardless of the absolute path of the folder, the scripts can run without problem.
        ```
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ```

Tasks Accomplished in Week 6:

A. Network Fine-tuning:
   - Continued training the network using the task-specific dataset to adapt it to our detection problem.
   - Monitored the training progress, analyzed intermediate results, and fine-tuned the model architecture and hyperparameters.

B. Evaluation Code:
   - Developed evaluation code for each stage of the model (encoder, RPN, object classification).
   - Integrated relevant metrics, such as precision, recall, and mAP, to measure the model's performance at different stages.

C. Data Visualization:
   - Implemented data visualization techniques to gain insights into the dataset and model predictions.
   - Visualized input images, ground truth annotations, and model predictions to validate the model's performance and identify potential issues.

D. Project Setup Procedure:
   - Documented a step-by-step project setup procedure to facilitate easy deployment on new machines or environments.
   - Included instructions for installing dependencies, configuring libraries, and setting up the necessary runtime environment.

By fine-tuning the network, implementing evaluation code, visualizing the data, and establishing a project setup procedure, we are ensuring the progress and maintainability of our project. These efforts allow us to continuously improve the model's performance, gain deeper insights into its behavior, and enable smooth deployment on different machines or environments. In the upcoming weeks, we will focus on further optimizing the model, conducting extensive evaluations, and preparing for the final model deployment.