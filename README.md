# Arvato Bertelsmann - Mail-Order Company Analysis 

This project is part of a challenge provided by Arvato Bertelsmann. The objective is to determine which individuals are more likely to become new customers of a mail-order company based in Germany. The analysis has three major phases: 
1. Part 0 - Get to Know the Data
* Exploratory analysis of the data from customers from the mail order company and the general population of Germany
* Data Cleaning according to documentation
* Data Scaling of features
* Principal component analysis to predict main components
2. Part 1 - Customer Segmentation
* K-Means Clustering for identifying the main clusters of both the general population and Germany. Base of this, a comparison was generated to check the relationship between both groups
3. Part 2 . Supervised Learning
* Imbalance classes correction for generating the model
* Test of machine learning models to see which model performs better.
* Code that generates the final prediction model

## Structure

```bash
├── code
│   ├── Part 0 - Get to Know the Data.ipynb
│   ├── Part 1 - Customer Segmentation.ipynb
│   ├── Part 2 - Supervised Learning Model.ipynb
│   └── scripts
│       └── preprocessing.py
└── files
    └── readme.txt
```

* Part 0 - Get to Know the Data.ipynb: Jupyter notebook with data preprocessing and analysis
* Part 1 - Customer Segmentation.ipynb: Jupyter notebook with customer segmentation
* Part 2 - Supervised Learning Model.ipynb: Jupyter notebook with prediction models
* preprocessing.py: Used for Part 1 and Part 2, this file contains all the preprocessing steps.

Note: The folder files has just a readme.txt describing the required files. The files need to be placed here first, before running the scripts. This files were omitted due to ownership restrictions.

### Prerequisites

You need to install the following:

1. Python
2. numpy
3. pandas
4. sklearn
5. matplotlib
6. notebook
7. imbalanced-learn


### Results

Check the results from the data analysis and model generation!

Blogpost: https://medium.com/@alejandro_garcia/mail-order-company-data-analysis-e22211623b2
