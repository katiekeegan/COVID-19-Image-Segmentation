# COVID-19 Image Segmentation
 
## Preprocessing the Data
# 1. Downloading the data
- Go to the [data website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742#80969742bcab02c187174a288dbcbf95d26179e8)
- Download the NBIA Data Retriever
- Download the Images (DICOM, 11GB), Annotations (JSON), and Clinical data (csv) into the `data` folder
- Use the NBIA Data Retriever to unzip the images (select 'Descriptive Directory Name' type)
- Run `code/preprocessing.py` to obtain the training and testing data (`x_val.joblib`,`y_val.joblib`,`x_train.joblib`,`y_train.joblib`)

# 2. Machine Learning Algorithms
# 2.1 Random Forest
- Run `code/rf_training.py`
- Note that `rf_training.py` generates the `rf_results.csv` file in the `results` folder, which is a spreadsheet that records the Dice coefficient for different 'n_estimators'. Feel free to change the `n_estimators` parameter, as well as the `n_jobs` parameters based on your computational resources.

# 2.2 SVM
- Run `code/svr_training.py`
- Note that `svr_training.py` generates the `svr_results.csv` file in the `results` folder, which is a spreadsheet that records the Dice coefficient for different `C` (we only use a linear kernel for this experiment with just the `C` parameter being tuned). Feel free to change the `C` parameter.

# 3. U-net
- Run `code/unet_training.py'
