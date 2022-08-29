# LinearRegression

This project will be your first steps into AI and Machine Learning. You're going to start with a simple, basic machine learning algorithm. You will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

The [first program](https://github.com/alfir-v10/LinearRegression/blob/main/train_model.py) will be used to predict the price of a car for a given mileage.

```bash
python estimatePrice -mileage 1000
```

$estimatePrice(mileage) = \theta_0 + \theta_1 * mileage$

The [second program](https://github.com/alfir-v10/LinearRegression/blob/main/estimatePrice.py) will be used to train your model. It will read your dataset file
and perform a linear regression on the data.

$$tmp\theta_0 = learningRate * \frac{1}{m} * \sum_{i=0}^{m-1} estimatePrice(mileage[i] - price[i])) $$

$$tmp\theta_1 = learningRate * \frac{1}{m} * \sum_{i=0}^{m-1} (estimatePrice(mileage[i] - price[i]))) * mileage[i] $$


```bash
python train_model --i data.csv --lr 0.01 --e 5000 --o weights.txt --p False --es 100.0
```
Argumetns:

* --i, help='input file', type=str, default='data.csv'
* --lr, help='learning rate', type=float, default=0.01
* --e, help='epochs', type=int, default=5000
* --o, help='output file', type=str, default='weights.txt'
* --p, help='plot results', type=bool, default=False
* --es, help='early_stopping', type=float, default=100.0

# Results
![](https://github.com/alfir-v10/LinearRegression/blob/main/mae.png)
![](https://github.com/alfir-v10/LinearRegression/blob/main/mse.png)
![](https://github.com/alfir-v10/LinearRegression/blob/main/result.png)
