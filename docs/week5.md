# Week 5: Encoder Training, Loss Testing, Hyperparameter Adjustment, and Network Development

During the fourth week of the project, we made significant progress in developing a workable code structure for the TPU platform and establishing a connection with Google Cloud Bucket. We also settled on using a pretraining technique and a two-stage model for detection. In the fifth week, our primary focus was on the training of the encoder, testing different loss functions, adjusting hyperparameters, and advancing the development of the remaining parts of the network. Additionally, we finalized our project file structure based on our growing insights and familiarity with the platform. This document provides an overview of our progress and outlines the tasks accomplished during Week 5.

1. Encoder Training:
   - We dedicated this week to training the encoder using the selected pretraining dataset. The encoder plays a crucial role in feature extraction, which forms the backbone of our detection model.

2. Loss Testing:
   - We conducted experiments with various loss functions to identify the most suitable option for our detection task.
   - We finalised on contrasive learning to do the self-supervised learning, the model will learn the robust features from different augmentated pairs and distinguish them from other pairs. It is show very effective in Goggle SimCLR and utilize unlabelled data.
   ```
   # Our custom loss according to relative papers
   # tau is a hyperparameter to be tuned, it is set to 0.05 since
   # the differences is very subtle in mammograms
   def NT_Xent_loss(a, b):
    tau = 0.05
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()
                           (a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)
   ```

3. Hyperparameter Adjustment:
   - Throughout the week, we carefully adjusted the hyperparameters of our model to improve its performance. This involved tuning parameters such as learning rate, batch size, regularization techniques, and optimizer settings. Our goal was to find optimal hyperparameter values that lead to faster convergence and better generalization.
   ```
   def print_iteration_stats(iter, train_loss, val_loss, n_iters, time_per_n_iters):
    print("Iter:{:5d}  |  Tr_loss: {:.4f}  |  Val_loss: {:.4f}  |  Time per {} iter: {}".format(
        iter, train_loss, val_loss, n_iters, convert_seconds_to_time(time_per_n_iters)))
   ```

4. Network Modularisation:
   - While training the encoder, we also started working on the development of the network training code structure. The aim is to make the code robust, easy to maintain and extend different functions.
   - We developed a custom library named **Mammolibs**, which include:
    - **MMdataset**: For custom dataset and basic data standardisation.
    - **MMmodels**: All the custom model structures and loss functions.
    - **MMutils**: All the utility functions such as printing loss or saving models.
    

5. Finalizing Project File Structure:
   - With an increasing understanding of the entire project and growing familiarity with the platform, we finalized our project's file structure. This involved organizing the code, data, and model checkpoints into well-defined directories and modules. A clear and organized file structure allows for easier maintenance, collaboration, and scalability as the project progresses.

Tasks Accomplished in Week 5:

A. Encoder Training:
   - Set up training pipeline for the encoder using the pretraining dataset.
   - Monitored training progress, analyzed intermediate results, and ensured convergence.

B. Loss Testing:
   - Implemented different loss functions and evaluated their impact on model performance.
   - Analyzed the trade-offs between precision and recall and selected the most appropriate loss function for our detection task.

C. Hyperparameter Adjustment:
   - Conducted systematic hyperparameter tuning experiments to optimize the model's performance.
   - Adjusted learning rate, batch size, regularization techniques, and optimizer settings based on experimental results.

D. Network Development:
   - Implemented the region proposal network (RPN) and object classification components.
   - Connected these components with the encoder to establish a complete two-stage detection model.

E. Finalizing Project File Structure:
   - Organized the project's code, data, and model checkpoints into a well-structured file hierarchy.
   - Ensured clear separation of concerns, modularity, and easy navigation within the project directory.

By completing the encoder training, testing various loss functions, adjusting hyperparameters, and progressing with the network development, we have made significant strides toward building an accurate and efficient detection model. Additionally, finalizing the project file structure allows for better organization and future scalability. In the upcoming weeks, we will focus on fine-tuning the model, conducting comprehensive