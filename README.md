# My Undergraduate and Graduate Projects

The author of this repo is Futian Zhang (Felix).

This is the repo that contains all my undergraduate projects. Here is the list and introduction to each of them.

List of Projects:  
Course Name: Project Name  
CSE258: Predicting Whether a Review is Funny on Steam
CSE256: Image Caption with GIT and modified mPLUG
CSE251B: Recognize Blurry Image with Transfer Learning
CSE151B: Trajectory Prediction using Repeat-Layer Linear Neural Network  
COGS108: Moba Game Winrate Prediction using SVM   
COGS118A: An Empirical Comparison of Supervised Learning Algorithms    
COGS118B: Rating Prediction using Unsupervised Learning Algorithms   


## Introduction of Projects:

### CSE258: Predicting Whether a Review is Funny on Steam
Report: [Link](https://github.com/FelixZhang1999/My_Undergrad_Project/blob/main/CSE256_Final_Project.pdf)
Summary: 
This project is the final project of CSE 258. In this project, I wrote models to determine whether a user review is funny or not.

The best model is RoBERTa+FC. RoBERTa is a pre-trained BERT-based model to convert text into latent spaces. Then the latent spaces are connected to a fully connected neural network for binary classification.

I finished this project solely.

### CSE256: Image Caption with GIT and modified mPLUG
Report: [Link](https://github.com/FelixZhang1999/My_Undergrad_Project/blob/main/CSE256_Final_Project.pdf)
Summary: 
This project is the final project of CSE 256. In this project, I wrote models to generate captions for images.

The best model is mPLUG from (this paper)[https://arxiv.org/abs/2205.12005]. mPLUG combines image and text processing using a shared Transformer backbone to capture the rich cross-modal representations.

I finished this project solely.

### CSE251B: Recognize Blurry Image with Transfer Learning
Code: [Link](https://github.com/f6zhang/CSE251_Final)
Report: [Link](https://github.com/millrogue/My_Undergrad_Project/blob/main/CSE251B_Final_Project.pdf)
Summary:
This project is the final report of CSE251B. In this project, we attempted to restore blurry images. 

Our best model uses a modified UNet and uses transfer learning from a large dataset. The result shows that our model can successfully restore blurry images, with a low MSE score between the original image and the output of the model, and with a higher classification accuracy when we apply a simple classification model on the output than on the original image.

I was crucially involved in all aspects of this project. I came up with the model structure and wrote a large portion of the code.

### CSE151B: Trajectory Prediction using Repeat-Layer Linear Neural Network 
Code: [Link](https://github.com/f6zhang/CSE-151B-Final-Project)
Report: [Link](https://github.com/millrogue/My_Undergrad_Project/tree/main/CSE151B:%20Trajectory%20Prediction%20using%20Repeat-Layer%20Linear%20Neural%20Network)
Summary:  
This project is a group class Kaggle Challenge (link)[https://www.kaggle.com/c/cse151b-spring# Global competition: https://eval.ai/web/challenges/challenge-page/454/overview]. In this project, we attempted to solve the problem of Trajectory Prediction for autonomous driving. The goal is to predict the trajectory of the vehicle 3 seconds in the future using 2 seconds data in the past. Our group (Two Three Three Three) scored first place in the public leaderboard with a test RMSE of less than 2.0 and scored second place in the private leaderboard.

Our group successfully developed two different neural network models to solve this problem. The first one is a repeat-layer linear model. This model is invented by me. It repeatedly uses linear layers in the model to minimize the influence of vanishing gradients. The second model is the encoder-decoder model. For more details, please check the project code and report.

I was crucially involved in all aspects of this project. I wrote the initial codebase, including loading and preparing the data, training, validating, and testing the model. I proposed and wrote our first model, the repeat-layer linear model. I was also responsible for hyperparameter tuning for both models. And I was responsible for writing the majority of the report.


### COGS108: Moba Game Winrate Prediction using SVM   
Code/Report: [Link](https://github.com/millrogue/COGS108_Final_Project)
Summary:  
This is a group class project. The goal is to successfully predict the outcome of a match in a MOBA game, Dota2, given the statistics of 5/10/15 minutes of the game.

We solved the problem using an SVM model and carefully selected input data. Our model's test accuracy (f1-score) is 75%/77%/78% given the statistics of 5/10/15 minutes of the game.

I was crucially involved in all aspects of this project. I finish this project solely. I was responsible for writing almost all the code and text content.

### COGS118A: An Empirical Comparison of Supervised Learning Algorithms   
Report: [Link](https://github.com/millrogue/My_Undergrad_Project/tree/main/COGS118A:%20An%20Empirical%20Comparison%20of%20Supervised%20Learning%20Algorithms) 
Summary:   
This is a solo class project. This paper is a replication of the previous work of A Empirical Comparison of Supervised Learning Algorithms:  
Caruana R, Niculescu-Mizil A (2006) An empirical comparison of supervised learning algorithms. In: Proceedings of the 23rd international conference on machine learning, ICML ’06. ACM, New York, NY, USA, pp 161–168.  

In this paper, I evaluate the accuracy of three algorithms, SVM, logistic regression, and random forest across three problems provided by the UCI Repository. 

### COGS118B: Rating Prediction using Unsupervised Learning Algorithms   
Code: [Link](https://github.com/FeiYin99/CCXXXIII---COGS-118B-Final-Project)
Report: [Link](https://github.com/millrogue/My_Undergrad_Project/tree/main/COGS118B:%20Rating%20Prediction%20using%20Unsupervised%20Learning%20Algorithms)  
Summary:   
This is a group class project. The goal is to use unsupervised learning methods to  make an anime recommendation system that predicts each user’s ratings on new anime that the user has not watched before. We used three methods: two types of PCA and a k-mean clustering.

I was responsible for the code and report for our first method: Anime-space PCA.
