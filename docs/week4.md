# Week 4: Pretraining and Two-Stage Model Implementation

During the third week of the project, we made significant progress in data collection, preprocessing, and initial model implementation. We gained valuable insights into the project and decided to leverage pretraining techniques and implement a two-stage model for detection. In the fourth week, our primary focus will be on developing a workable code structure specifically designed for the TPU platform and establishing a connection with Google Cloud Bucket. This document provides an overview of our progress and outlines the tasks for Week 4.

### 1. Pretraining Technique Selection:
   - After careful consideration and analysis, we have decided to employ pretraining techniques for our model.
   - Generative models were rejected because they require huge amount of high quality training data and the generated images will be lack of validation, so the truthfulness of the data cannot be guaranteed.
   - Transformer based model were also rejected because they often require much more training time and data. Additionally, they are not as accessible as CNN models and the improvement of them over tranditional CNN model is very slight and cannot guarantee.
   - We use different techniques like data augmentation to enrich the dataset while ensuring the truthfulness. The unlabelled data can be well utilized to improve model performance by self-supervised learning.

### 2. Two-Stage Model for Detection:
   - We have determined that a two-stage model architecture would be suitable for our detection task. This approach involves separating the detection process into two stages: region proposal and object classification. It has shown promising results in similar tasks and can help us achieve accurate and efficient detection.
   - The model pipeline we agreed on will be using U-Net for segmentation(finding ROIs) and ResNet for local classification. Since these models were matured enough and proved their effectiveness in various tasks.

### 3. TPU Platform Code Structure:
   - To leverage the power of Google Cloud's TPU platform, we need to develop a code structure that is optimized for TPU hardware. This structure should efficiently distribute computation across TPUs, take advantage of parallel processing, and maximize resource utilization.
   - TPUs use very different training libraries compare to training on CPU or CUDA devices. They are also not very well documented and very can be very differently used when training on TPU VMs in GCP or other TPUs such as Colab.

    ```
    # We use xla library for training on TPUs
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

    def train_encoder(index, ...):

        # XLA dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(...)
        train_loader = torch.utils.data.DataLoader(...)

        ...

        # Model definition and transfer learning
        model = MMmodels.Pretrain_Encoder()
        if state_dict:
            model.load_state_dict(state_dict)
        model = model.to(device).train()

        ...

        # Main training loop
        for it in range(...):
            para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

            for batch in para_train_loader: # noqa
                images, labels = batch
                logits = model(images)
                train_loss = criterion(logits, labels)
                optimizer.zero_grad()
                train_loss.backward()
                xm.optimizer_step(optimizer)

        # Tr/Val loss and save model
        ...

    if __name__ == '__main__':
        ...

        # Training with TPUs
        xmp.spawn(train_encoder, args=(
        state_dict,     # model
        dataset,        # dataset
        lr,             # lr
        pre_iter,       # pre_iter
        n_iter,         # niters
        64,             # batch_size
        current_dir,    # current_dir
        args.test
        ), start_method='forkserver')
    ```

### 4. Google Cloud Bucket Integration:
   - We will establish a connection with Google Cloud Bucket to facilitate seamless data storage and retrieval. This integration will enable us to efficiently access and process our datasets, as well as store the trained models and other project artifacts.

    ```
    # We built our own dataset for GCP cloud storge buckets
    class MMImageSet(Dataset):
        def __init__(self, gcs_path, stage='encoder', aug=True):
            super(MMImageSet, self).__init__()
            self.fs = gcsfs.GCSFileSystem()
            self.stage = stage
            self.filenames = [...]
    ```

## Work Summary
By focusing on developing a workable code structure for the TPU platform and establishing a connection with Google Cloud Bucket, we aim to lay the foundation for efficient training and inference in subsequent stages of the project. This will enable us to leverage the power of distributed computing and seamlessly manage our data and models in the cloud environment.