Project Abstract:
This project focuses on the implementation of several artificial neural network models on a dataset of COVID-19 cases from
India. The main goal is to evaluate these methods on long and short term training periods to find one that can efficiently predict
the numbers in both cases while retaining a high efficacy. The project is based on the study "Multiple-Input Deep CNN Model
for COVID-19 Forecasting in China" by Huang et al. 1 , that focuses on the need of an artificial neural network model capable of
forecasting COVID-19 cases with only a short time period as training data at hand during the start of the COVID-19 pandemic.
For this purpose, the artificial neural network methods Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Long
Short-Term Memory (LSTM), and Gate Recurrent Unit (GRU) were used for predicting the cumulative confirmed cases of a
given day based on the data of the previous five days including six time sequences (features) that influence the confirmed
cases. The methods were evaluated with the R 2 score, Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
metrics using multiple features as input. Like this, we also implemented the aforementioned methods using the Keras and
Pytorch Python libraries and evaluated them respectively. The Indian dataset provides the same features and the models were
trained and tested on data from five Indian states. Our experimental results revealed that the deep CNN model shows the best
efficacy and the lowest error. We also explore the data from India in terms of health care, population density and interventions
in the respective states that were most affected.

# Data-Science-in-Life-Science-Project
This project containts three folder:
1. Implementation folder: It contains the implementation of project in Keras and pytorch both. However, we are using pytorch for final results.
To run the code you need to put the code on google drive in single folder "colab folder" and then open the files in google colab. You require the following pytorch version to execute the code: 1.9.0+cu102. You need the GPU as well to run the code. 
Other required information can be found in "read me" file at Implementation Folder.

2. Indian-States-Covid19-Datasets Folder: It contains the data of all states.
4. Indian-States-Model-Results: The results of CNN, LSTM, MLP and GRU deep learning models can be foound in this folder.

Key Results:
Proposed CNN model is the best model comparing to other models irrespective of data.

Long Data: March 10, 2020 to June 30, 2021

Short Data: March 10, 2020 to June 18, 2020


Result on Short Data:

![Screenshot from 2021-08-09 15-13-27](https://user-images.githubusercontent.com/77930296/128712070-192af4e0-f7f9-4f5a-9c41-95409f096a89.png)


Mean Absolute error:

![Mean-Absolute-Error-on-short-data](https://user-images.githubusercontent.com/77930296/128711124-320b1622-f40f-4393-96bb-996d6b26556f.png)

Root Mean Squared Error:

![Root-mean-squared-error-on-short-data](https://user-images.githubusercontent.com/77930296/128711485-d93fbd1d-037c-4b8a-bc28-61ab0c91e723.png)

  
R2 Score:

![R2-score-on-short-data](https://user-images.githubusercontent.com/77930296/128711409-a2e5cc94-c051-4862-995a-6c4c8ca524b6.png)

Result on Long Data:

![Screenshot from 2021-08-09 15-13-41](https://user-images.githubusercontent.com/77930296/128712131-5ac438fa-5c74-4ae6-a17d-7657923bf6ef.png)


Mean Absolute error:

  ![Mean-Absolute-Error-on-long-data](https://user-images.githubusercontent.com/77930296/128711296-f4817b3f-c25d-4ff0-9842-2a9d93228a7f.png)

Root Mean Squared Error:

  ![Root-mean-squared-error-on-long-data](https://user-images.githubusercontent.com/77930296/128711372-e24f511c-0b41-4f3d-ba8d-6a6591db27a7.png)

R2 Score:

  ![R2-score-on-long-data](https://user-images.githubusercontent.com/77930296/128711343-87a9e24e-679e-40b2-ba0c-06f3dca47154.png)

