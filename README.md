Project Proposal: Used Car Price Estimator with Recommendation System
By: Lukas Chang ldc105, Daniel Li dal279

Project Definition:
Problem Statement
The used car market is a vast and often complex landscape, with prices varying significantly based on a multitude of factors such as the car’s age, mileage, make, model, condition, transmission type, and more. Sellers and buyers often face challenges in determining fair market value due to fluctuating market trends, regional differences, and the subjective nature of car conditions. This project aims to develop a data-driven solution that predicts the market price of a used car based on these key attributes using a robust prediction model. In addition, the system will offer recommendations for similar cars, providing a comprehensive view for consumers and businesses alike.

Strategic Aspects
The project involves strategic aspects such as:
Developing a reliable and accurate prediction model based on historical sales data of used cars.
Efficiently managing a large dataset using SQL to store, query, and preprocess data.
Implementing data preprocessing, feature engineering, model training, and performance evaluation using Python.
Creating a recommendation system to enhance the user experience by suggesting similar cars based on specified criteria.
Offering insights that can help users make informed decisions about buying or selling used cars.
This project relates closely to key data management concepts discussed in lectures, such as regression analysis, data preprocessing, feature engineering, data management practices, and model evaluation. It also incorporates database management concepts, aligning with lectures on pandas, numpy, SQL optimization, data models, data queries, and structured data storage.

Novelty and Importance:
Importance
As people who have shopped for cars, we understand how difficult it can be to find the right car within the price range. Many factors influence the price of a vehicle, and with fluctuating prices and varying consumer preferences, it's hard to know what price is right. Having a reliable and accurate data-driven price prediction model can provide its users with a clear understanding of what price they should be selling their car for or what the vehicle they should be searching for should be priced at. It empowers sellers to set competitive prices and maximize their returns while empowering buyers with critical information for better negotiation and informed decision-making. We’re excited to create a tool that we would actually use ourselves, and plus we both just love cars, and to work on a project that has to do with cars sounds really fun. 

Novelty
Data Integration: Combining large-scale historical data with predictive modeling to provide personalized value estimates.
Comprehensive Recommendations: Unlike existing pricing estimators, this project offers recommendations for similar cars based on user input, giving more context to the estimated price.
Data Management Optimization: Using SQL for efficient data storage and querying enhances the scalability and performance of handling large datasets.

Plan:
Dataset and Data Management:
In this project, we will work with a large dataset of used cars for sale, sourced from Kaggle called “Used Cars Dataset”. The dataset will include various features such as make, model, year, mileage, condition, transmission, fuel type, and price. These features will be key in training a predictive model that estimates car prices. For data management, SQL will be used to efficiently store and query the data. Python will be employed for data manipulation, analysis, and the development of machine learning models.

Data Preprocessing:
This phase involves cleaning, reducing, and preparing the data for analysis. Since the current database is 1.45 GB, we will shrink it by randomizing and selecting a fraction of the entries while preserving data integrity. We will handle missing values through imputation or removal, encode categorical features like transmission type using one-hot encoding, and scale numerical features like mileage and year for consistent modeling. Finally, the dataset will be split into training and testing sets for model development.

Machine Learning Model:
For the machine learning model, we will start with linear regression as a baseline to predict car prices based on the various features. Linear regression is simple and interpretable, allowing us to understand the relationships between features and the predicted price. We will also analyze feature importance to identify which attributes have the most significant influence on car prices. 

Recommendation System:
To enhance the user experience, a content-based recommendation system will be developed. This system will suggest similar cars based on the user’s input, such as make, model, price range, and other car specifications. By using content-based filtering, the system will identify cars with similar features and provide recommendations, making it easier for users to find alternatives.

Evaluation and Testing:
We will measure success using metrics like Mean Squared Error (MSE) and R² score to assess regression accuracy and model variance explanation. Cross-validation will ensure robustness by testing the model on different data subsets. The recommendation system’s performance will be evaluated based on the relevance and usefulness of suggestions, with user feedback aiding in fine-tuning.


