[My Linkedin Profile](https://www.linkedin.com/in/kspraharshita/)

[Download my Resume here](https://github.com/pk2971/Praharshita/files/8794154/Resume-.Praharshita.Kaithepalli.pdf)


Hello recruiter!  

I am a Data science, Masters student at Rochester Institute of Technology. I enjoy working with challenging and real-world datasets to derive meaningful insights and business-oriented solutions. I am looking for Summer/Fall 22' internships. These are some of the projects that I've worked on. 

# [Predicting AirBNB Prices using Regression Models-Python](https://github.com/pk2971/AirBNB-pricing-prediction-analysis/blob/main/AirBNB_pricing_prediction_analysis_Praharshita_Kaithepalli.ipynb)

- Created a project to predict the prices of AirBNB listings based off on various factors.
- Cleaned a data set with over 30,000 rows to make it available for training.
- Used various algorithms such as LinearRegressor, RandomForestRegressor, GradientBoostRegressor, XGBRegressor etc. to obtain the best performing model.
- MSE, RMSE and R^2 obtained to study the prediction accuracy for each algorithm
- Hyperparameter tuning using BayesianOptimizer performed on the best performing algorithm(XGBoost) to obtain better prediction scores.

```
#Before Hyper parameter tuning
print("\nMAE:", round(mean_absolute_error(predict_xgbr,y_test)))
print("\nMSE:", round(mean_squared_error(y_test,predict_xgbr),4))
print("\nRMSE:", round(math.sqrt(mean_squared_error(y_test, predict_xgbr)),4))
print("\nR^2:", round(r2_score(y_test,predict_xgbr),4))
MAE: 31

MSE: 1891.7184

RMSE: 43.4939

R^2: 0.6427

perform_scoring(predict_xgbr,y_test)

68.58646396725625

#After Hyperparameter Tuning
print("\nMAE:", round(mean_absolute_error(predict_xgbr,y_test)))
print("\nMSE:", round(mean_squared_error(y_test,predict_xgbr),4))
print("\nRMSE:", round(math.sqrt(mean_squared_error(y_test, predict_xgbr)),4))
print("\nR^2:", round(r2_score(y_test,predict_xgbr),4))
MAE: 29

MSE: 1760.6226

RMSE: 41.9598

R^2: 0.6675

perform_scoring(predict_xgbr,y_test)

71.15918725332554
```

# [Temperature Prediction using Air Quality using LSTM and Conv1D, Multivariate Time Series Forecasting-Python](https://github.com/pk2971/Air-Quality-vs-Temperature-time-series)

- Built a time series forcasting model to predict the atmospheric temperature at a given time of the day based off on the pollutant concentrations in the atmosphere.
- Cleaned a data set of 9000+ rows and prepared it to enable time series forecasting.
- Trained the data set on Conv1D and LSTM models and achieved high accuracy.
- Graphed the predicted vs. actual values of both data sets.

![timeseries](https://user-images.githubusercontent.com/89590898/170887795-d4900680-bbdf-4c9b-978f-471c0180e3c1.png)

```
test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
plt.plot(test_results['Test Predictions'][0:100])
plt.plot(test_results['Actuals'][0:100])
```
![predicted vs actuals](https://user-images.githubusercontent.com/89590898/170887927-2884c750-1d78-43b2-b711-d74ec5191607.png)


# [COVID-19 Image Classifier using VGG16-Python](https://github.com/pk2971/COVID-19-Image-Classification)

- Built a classifier to identify if a lung is effected by COVID-19 based on scan images.
- Used an ImageGenerator to scale images so that they can be efficiently used by the model.
- Used transfer learning on a CNN trained VGG16, which created time efficient and solid results.
- Predicted if the lung is effected by COVID-19 with 99% accuracy(from a new image).

![LungNormal](https://user-images.githubusercontent.com/89590898/170888118-a070a0e0-d5a9-40e6-9c83-36558415513a.jpeg)
```
import cv2
img=cv2.imread('/content/ChestCOVID19.jpg')
plt.imshow(img)
resize=tf.image.resize(img,(150,150))
plt.imshow(resize.numpy().astype(int))
yhat=model.predict(np.expand_dims(resize/255,0))
yhat
array([[0.99022114, 0.00555698, 0.00422185]], dtype=float32)
#Predicted with 99% probability that the lung is effected by Covid-19
```
# [Twitter Sentiment Mining using afinn and syuzhet-R language](https://github.com/pk2971/Twitter-setiment-mining)
- Extracted 2000 tweets from Twitter website about Fenty Beauty cosmetic company and stored them in data frames.
- Cleaned the tweet text data from mentions, emojis, profanities and various other symbols.
- Removed data in the tweeets not relating to the cosmetic company.
- Created wordclouds of different products to see what is being talked about the most.
- Performed setiment analysis and assigned sentiment score to tweets using afinn and syuzhet packages.
- Tableau used to further visualize the sentiment scores of the tweets for different products in a particular year.
- Simplified the findings to report it to a non-technical person.

# Other skills and softwares
- Programming Languages: Java, R, C, mySQL, SQL.
- Softwares: Tableau, MS Excel, MS Access, JMP statistical software, Minitab.

