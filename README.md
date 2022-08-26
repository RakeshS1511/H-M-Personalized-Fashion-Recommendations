# H-M-Personalized-Fashion-Recommendations
##  H&M’s online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are essential.
## The Data
### Dataset
We were given three datasets + 1 collection of image data.
1. Article (product) metadata. It consists of product name, categorical columns describing product’s colour, department, garment group, and descriptions.
2. Customer metadata. It consists of club member status, whether they subscribe to fashion news or not, and age.
3. Historical transactions.
4. Article images. We don’t use this in our solution, since we only use limited computation power.
## Pattern
The datasets consist of 106k products and 1.37m customers. If we included all products as candidates per customer, we’ll have 106k X 1.37mn =145 billion rows of data, not to mention the size of the features. We need to find a better way to reduce the data or we can’t fit the data into RAM.

One great analysis from Kaggler here helps us to focus on only relevant products for customers. Shown number of article count per week group by the last time it’s been bought before this week. The majority of transactions made by articles that's still been bought last 1 week. From this, we decide to only use articles that's still been bought last 6 weeks as candidates. Later we’ll filter these candidates again to reduce the number of candidates per customer.

Our solution was inspired by famous two steps architecture: (1) Candidate Retrieval and (2) Ranking. Candidate Retrieval focuses more on recall means that it aims to filter 106k products to 30 that are relevant for customers. Ranking focuses more on precision means that it aims to correct the order of 30 candidates and select only top 12 products.
![image](https://user-images.githubusercontent.com/108456495/185996878-cd8a84e2-e8d7-4fc5-a3e1-70fdf29d042f.png)
## Candidate Retrieval (Recall)
### Cold-start problem
Cold-start problem is a problem where new customer coming and we don’t have any information about them. We used 12 popular products from last 1 week to address this problem.

### Rule-based candidate retrieval
We used several strategies as our Recall model.
1. Popular products from last 1-week group by customer segment
2. Products that were previously purchased by customer
3. Products that are bought together
4. Products with similar price
5. TensorFlow Recommenders Retrieval

With these strategies, we have ~ 7% recall. There are possible techniques that we didn’t use: collaborative filtering, similarity from image embedding, and similarity from product description embedding.

## Ranking (Precision)
From start, we decided to develop ranking models separately. We aimed to have diverse models so that later we could ensemble the predictions. Our approach was different even from how we prepared the training data.

## Training Data
In the context of ranking ML, the training data must contain both positive and negative samples. Positive samples are actual purchased from historical transaction, we label it as 1 (purchased). Negative samples are sample that we label as 0 (not purchased). Why we need negative samples? Because we’ll use ranking ML not only for sorting products that customer bought but also products that customer not bought. Remember that our Candidate Retrieval only has 7% recall means most of candidates are not actually bought by customers. Then how we generate negative samples?

### Two techniques that we use for generating negative samples.
1. Use the output from Candidate Retrieval since it has both positive and negative samples.
2. Randomly select N available products as negative samples.

### Feature Engineering
In the context of ranking ML, there is 3 type of feature.
1. User feature. It’s derived only from customer data. Example: age, num_trx_last_90d, mean_price_last_90d, etc
2. Item feature. It’s derived only from product data. Example: product_colour, product_category, product_price, num_bought_last_7d, etc
3. User-Item feature. It’s derived from interaction between customer and product. Example: difference user_mean_price and product_price, how many user purchase product with the same colour/category/etc. It’s important to have good user-item feature to get well performing model.

We had varying windows of observation for feature engineering i.e. all time, 8/4/1 weeks. Following are features that we use
![image](https://user-images.githubusercontent.com/108456495/185997262-340cced1-4f84-4cbd-b8e6-87b19c85ba77.png)
## Learning Algorithms
Since we label the data with 1/0, we could use either Ranker algorithm or Classification algorithm . We mainly use 2 libraries
1. LGBM. (LGBMRanker & LGBMClassifier)
2. CatBoost. (CatBoostClassifier)

The final submission was an ensemble of 7 different models. We differentiate how we generate negative samples in training data with several positive and negative sample ratios hence we got quite diverse models in the end.
![image](https://user-images.githubusercontent.com/108456495/185997328-42f62cd3-37ad-4bf3-8b05-982e9ddc3a0b.png)

## Validation
It’s very important to set up robust validation strategy for Kaggle competition. We used last 7 days in the training data as our validation set. It would be better if we have > 1 fold validation set, but with computation and time constraints we decided to only have 1 fold validation set. We observed with only 1 fold validation set is still giving us good correlation with leaderboard score.

![image](https://user-images.githubusercontent.com/108456495/185997435-cbbd988e-759b-4cd7-bdc4-2559740001c4.png)


