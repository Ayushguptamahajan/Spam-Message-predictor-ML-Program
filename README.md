# Spam-Message-predictor-ML-Project
Spam Message detector Machine learning project (- end- to- end project)

### **The summary of the steps executed in ML model are:**

**1. Data collection and overview of the dataframe:**
- The data was extracted from csv file downloaded from kaggle.
- Data consist of 5572 rows and 5 columns.
- Intuition of the data was gained via calling the first five and last five rows.
- There were 99% Nan values present in Unnamed: 2, unnamed: 3 and unnamed: 4 features, so i decided to drop these features.

**2. Feature Engineering:**
- Since Number of character, word and sentences plays a major role in defining a message as spam or pam so i decided to construct such new features in dataframe in order to increase the accuracy and precision of the model.**(Nltk library was used for such construction of feature)** 
- In any sentence/message punctuation, stopword and special character doesnot play any role for deciding factor of spam or ham. So the same text message was changed to a new feature by removing punctuation mark, stopword and special character. Thereafter stemming of each word in message was done to obtain the transformed_text. 

**WordCloud was created for both the ham and spam categories** to get visualisation of words used in ham and spam messages

**3. Exploratory Data Analysis:**
- Univariate Analysis:
    - Dist and QQ plot was ploted for numerical columns which depicted none of the numerical columns are normally distributed. Reason being presence of outliers.

- Bivariate and Multivariate Analysis:
    - Pie plot was created for both the ham and spam categories which indicated that 12% data was spam. **(Highly imbalanced dataframe)**
    - Out of curiosity, self has found the top 30 ham and spam messages word used in the given data set. **(Countplot was plotted for the same)**
    - Function was created for plotting the hist plot of the numerical columns with hue factor (spam and ham). Graph implied spam text has less no of character and word.

**Further pairplot and heatmap was ploted** - Result depicted strong relation of target feature with the number of character compared to other numerical features in the resultant dataframe.

**4. Vectorization of text and target column:**
In case of text feature vectorization need to be performed for accurate result and precision score.

**5. Model Building:**

- Train test split was performed with test_size as 20% and random state of 0.2.
- Further **eleven** model was fitted for the splitted data set and various accuracy_score/prescission score was fetched. 

**Name of model used are below:**
   - LogisticRegression
   - GaussianNB
   - MultinomialNB
   - BernoulliNB
   - KNeighborsClassifier
   - DecisionTreeClassifier
   - RandomForestClassifier
   - BaggingClassifier
   - ExtraTreesClassifier
   - XGBClassifier
   - AdaBoostClassifier
   - SVC
    
**6. Conclusion:**
- Since the given problem need to have **type 1 error problem**, so we need to select the model which give high precision with feasible accuarcy.
- **MultinomialNB algorithm** outstanded the performance of model with **accuracy_score of ~95%** and **precision_score of 1.00**.So i selected **MultinomialNB algorithm** of **naive_bayes with tgifvectorization** as the final model for the above said problem.
