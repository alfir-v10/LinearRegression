# LinearRegression

This project will be your first steps into AI and Machine Learning. You're going to start with a simple, basic machine learning algorithm. You will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

The first program will be used to predict the price of a car for a given mileage.

$estimatePrice(mileage) = \theta_0 + \theta_1 * mileage$

The second program will be used to train your model. It will read your dataset file
and perform a linear regression on the data.

$tmp\theta_0 = learningRate * 1 / m * \sum_{i=0}^{m-1}estimatePrice(mileage[i] - price[i])$
