# Dataset Enhancement
The objective of this section is to augment the dataset for our machine learning model, aiming to increase the quantity of images available for training. 
Data augmentation is crucial to ensure an adequate amount of data for model training. 
Additionally, it helps mitigate the risk of overfitting, ensuring that the model generalizes well to unseen data.

## Datasets Included
| Name | Images | Included | Labelled |
| --- | :---: | :---: | :---: | 
| [CBIS-DDSM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad) | 10239 | :heavy_check_mark: | :heavy_check_mark: |
| [InBreast](https://www.kaggle.com/datasets/martholi/inbreast?select=inbreast.tgz) | 410 | :heavy_check_mark: | :x: |
| [MIAS](http://peipa.essex.ac.uk/info/mias.html) | 322 | :heavy_check_mark: | :heavy_check_mark: |
| [Breast-Cancer-Screening-DBT](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580#6468558050a1e1bdf0de46de92128576e1d3e9b1) | 22032 | :x: | :grey_question: |
| [BCDR](https://bcdr.eu/information/downloads) | 956 | :x: | :grey_question: |
| [CDD-CESM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611) | 2006 | :x: |  :grey_question: |
| [CMMD](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508) | 5202 | :heavy_check_mark: | :x: |
| [Duke Breast Cancer MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) | 773888 | :x: | :grey_question: |
| [King Abdulaziz](https://www.mdpi.com/2306-5729/6/11/111#) | 2378 | :heavy_check_mark: | :heavy_check_mark: |
| [Embed](https://pubs.rsna.org/doi/10.1148/ryai.220047) | 68000 | :heavy_check_mark: | :heavy_check_mark: |
| [OMI-DB](https://www.cancerresearchhorizons.com/licensing-opportunities/optimam-mammography-image-database-omi-db) | 2620 | :x: | :grey_question: |

Reasons for Exclusion of Certain Datasets as of June 6, 2023:

1. Dataset Selection Criteria:
In selecting the datasets to include in our study, we encountered limitations regarding the availability and suitability of certain datasets. 
Our focus was specifically on 2D breast mammographies, which are the most commonly used and cost-effective method for breast cancer detection<sup>[1]</sup>. 
Although 3D mammographies have shown potential for enhanced cancer detection, their widespread implementation and higher cost pose challenges for broader adoption in healthcare institutions. 
Hence, to ensure consistency and practicality, we made the decision to exclusively consider datasets containing 2D mammography images. 
Consequently, the following datasets were not included in our analysis: Breast-Cancer-Screening-DBT, CDD-CESM, and Duke Breast Cancer MRI.

2. Pending Data Access Requests:
In some cases, datasets of interest were not publicly available, necessitating our team to submit access requests to the respective data issuers. 
As of the current report, we are awaiting responses from the data issuers for the BCDR and OMI-DB datasets. 
Once the necessary permissions are obtained, we will include these datasets in our study and conduct the required analyses accordingly.

These decisions were made based on careful consideration of the research objectives, available resources, and the most relevant and accessible datasets within the scope of the project.

## Data Extraction
Data extraction for the CBIS-DDSM and CMMD datasets involved utilizing the NBIA Data Retriever tool<sup>[3]</sup> to download the DICOM images. These images were then transferred to a Google Virtual Machine (VM) with a storage capacity of up to 5 TB. The usage of the Google VM ensured efficient storage and management of the large-sized datasets.

In contrast, the InBreast, MIAS, and KingAbdulaziz datasets, being considerably smaller in size, were downloaded directly onto our local machines for further processing and analysis.

These approaches were employed to accommodate the varying sizes of the datasets and optimize the data handling process accordingly.

## Data Augmentation
Upon gathering all the datasets, the following augmentations were applied:

1. Rotation: To ensure the model's ability to identify patterns rather than rely on the specific location of Regions of Interest (ROIs) within the 2D breast mammography images, rotation augmentation was implemented. 
Specifically, the images were rotated by 30 degrees in 11 different angles. 
This technique aids in training the model to recognize the patterns indicative of ROIs.

2. Contrast Limited Adaptive Histogram Equalization (CLAHE): CLAHE, a widely used technique in medical image processing, including breast mammography, was employed to enhance contrast and improve the visibility of details within the images. 
This technique helps in accurately detecting and diagnosing breast abnormalities by highlighting structures and enhancing image details. 
In cases where the presence of ROIs is uncertain, scaling the contrast of the mammography images assists medical professionals in identifying these areas.
By replicating this technique in our model, we aim to equip our machine learning algorithm with the ability to enhance the contrast and improve the visibility of key structures, enabling it to better identify potential ROIs for suspicious regions within breast mammograms.

3. Resizing: Due to variations in image dimensions and aspect ratios among the datasets, resizing was necessary to normalize the images for feeding into the machine learning model. 
Initially, a size of 1024 x 1024 pixels was selected for all images. 
However, considering the large number of parameters involved and the substantial dataset size (over 120,000 images for each labeled and unlabeled dataset after augmentation), it was determined that downsizing the images to 256 x 256 pixels would be more manageable for the model while maintaining relevant information.

These data augmentation techniques were implemented to enhance the quality and consistency of the dataset, thereby improving the performance and generalization capabilities of the machine learning model during training and evaluation.

## Data Upload
Once the labeled and unlabeled datasets were combined and prepared for further analysis, the next step involved uploading the datasets into a Google Buckets folder. 
Google Buckets provide a scalable and reliable cloud storage solution, facilitating easy accessibility and management of the data.

## References
[1] Cancer Research UK (2023) [https://www.cancerresearchuk.org/about-cancer/breast-cancer/getting-diagnosed/tests](https://www.cancerresearchuk.org/about-cancer/breast-cancer/getting-diagnosed/tests)

[2] Harrington Hospital (2023) [https://www.harringtonhospital.org/women_blog/comparing-2d-3d-mammography/#:~:text=Although%20a%203D%20mammogram%20is,3D%20mammogram%20in%20most%20women.](https://www.harringtonhospital.org/women_blog/comparing-2d-3d-mammography/#:~:text=Although%20a%203D%20mammogram%20is,3D%20mammogram%20in%20most%20women.)

[3] National Cancer Institute (2023) [https://wiki.nci.nih.gov/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide](https://wiki.nci.nih.gov/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide)
