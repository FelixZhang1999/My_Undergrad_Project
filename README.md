# My Undergraduate Projects

This author of this repo is Futian Zhang (Felix).

This is the repo that contains all my undergraduate projects. Here is the list and introduction to each of them.

List of Projects:  
Course Name: Project Name  
CSE151B: Trajectory Prediction using Repeat-Layer Linear Neural Network  
COGS108: Moba Game Winrate Prediction using SVM   
COGS118A: An Empirical Comparison of Supervised Learning Algorithms    
COGS118B: Rating Prediction using Unsupervised Learning Algorithms   

## Introduction of Projects:

### CSE151B: Trajectory Prediction using Repeat-Layer Linear Neural Network 
Code: https://github.com/f6zhang/CSE-151B-Final-Project   
Report: https://github.com/millrogue/My_Undergrad_Project/blob/main/CSE151B:%20Trajectory%20Prediction%20using%20Repeat-Layer%20Linear%20Neural%20Network/CSE151B_Project_Final_Report.pdf

Summary:
This project is a group class Kaggle Challenge (link: https://www.kaggle.com/c/cse151b-spring# Global competition: https://eval.ai/web/challenges/challenge-page/454/overview). In this project, we attemptted to solve the problem of Trajectory Prediction for autonomous driving. The goal is to predict the trajectory of the vehicle 3 seconds in the future using 2 seconds data in the past. Our group (Two Three Three Three) scored first place in public leaderboard with a test RMSE of less than 2.0 and scored second place in private leaderboard. (Before we presented our model in class, we were first place. And the second place team modified their model after our presentation and got higher on private leaderboard).

Our group successfully developed two different neural network models to solve this problem. The first one is a repeat-layer linear model. This model is invented by me. It repeatly uses linear layers in the model to minimize the influence of vanishing gradient. The second model is encoder decoder model. For more details, please check the project code and report.

I crucially involved in all aspects of this project. I was responsible for writing the initial codebase, including loading and preparing the data, training, validating and testing the model. I proposed and wrote our first model, repeat-layer linear model. I was also responsible for hyperparameters tuning for both models. And I was responsible for writing the majority of the report.


### COGS108: Moba Game Winrate Prediction using SVM   
Code/Report: https://github.com/millrogue/COGS108_Final_Project   
Summary:  
This is a group class project. The goal is to successfully predict the outcome of a match in a moba game, Dota2, given the statistics of 5/10/15 minutes of the game.

We solved the problem using a SVM model and carefully selected input data. Our test accuracy (f1-score) of our model is 75%/77%/78% given the statistics of 5/10/15 minutes of the game.

I crucially involved in all aspects of this project. I basically finish this project solely. I was responsible for writing almost all the code and text content.

### COGS118A: An Empirical Comparison of Supervised Learning Algorithms   
Code/Report: https://github.com/millrogue/My_Undergrad_Project/tree/main/COGS118A:%20An%20Empirical%20Comparison%20of%20Supervised%20Learning%20Algorithms    
Summary:   
This is a solo class project. This paper is a replication of previous work of A Empirical Comparison of Supervised Learning Algorithms:  
Caruana R, Niculescu-Mizil A (2006) An empirical comparison of supervised learning algorithms. In: Proceedings of the 23rd international conference on machine learning, ICML ’06. ACM, New York, NY, USA, pp 161–168.  

In this paper, I evaluate the accuracy of three algorithms, SVM, logistic regression, and random forest across three problems provided by the UCI Repository. 

### COGS118B: Rating Prediction using Unsupervised Learning Algorithms   
Code: https://github.com/FeiYin99/CCXXXIII---COGS-118B-Final-Project   
Report: https://github.com/millrogue/My_Undergrad_Project/tree/main/COGS118B:%20Rating%20Prediction%20using%20Unsupervised%20Learning%20Algorithms   
Summary:   
This is a group class project. The goal is to use unsupervised learning methods to  make an anime recommendation system that predicts each user’s ratings on new anime that the user has not watched before. We used three methods: two types of PCA and a k-mean clustering.

I was responsible for the code and report for our first method: Anime-space PCA.
