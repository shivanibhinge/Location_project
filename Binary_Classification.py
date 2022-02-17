#!/usr/bin/env python
# coding: utf-8

# #  <span style="color:orange">Binary Classification Tutorial (CLF101) - Level Beginner</span>

# In[29]:


import pandas as pd
dataset = pd.read_csv('Location_data_final_.csv')


# In[30]:


#check the shape of data
dataset.shape


# In[31]:


data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[32]:


from pycaret.classification import *


# In[33]:


exp_clf101 = setup(data = data, target = 'anomaly', session_id=123) 


# # 7.0 Comparing All Models

# In[34]:


best_model = compare_models()


# In[35]:


print(best_model)


# # 8.0 Create a Model

# In[36]:


models()


# ### 8.1 Decision Tree Classifier

# In[37]:


dt = create_model('dt')


# In[38]:


#trained model object is stored in the variable 'dt'. 
print(dt)


# ### 8.3 Random Forest Classifier

# In[20]:


rf = create_model('rf')


# Notice that the mean score of all models matches with the score printed in `compare_models()`. This is because the metrics printed in the `compare_models()` score grid are the average scores across all CV folds. Similar to `compare_models()`, if you want to change the fold parameter from the default value of 10 to a different value then you can use the `fold` parameter. For Example: `create_model('dt', fold = 5)` will create a Decision Tree Classifier using 5 fold stratified CV.

# # 9.0 Tune a Model

# ### 9.1 Decision Tree Classifier

# In[39]:


tuned_dt = tune_model(dt)


# In[40]:


#tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)


# ### 9.2 K Neighbors Classifier

# In[18]:


import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})


# In[19]:


print(tuned_knn)


# ### 10.1 AUC Plot

# In[24]:


plot_model(tuned_dt, plot = 'auc')


# ### 10.2 Precision-Recall Curve

# In[29]:


plot_model(tuned_dt, plot = 'pr')


# ### 10.3 Feature Importance Plot

# ### 10.4 Confusion Matrix

# In[30]:


plot_model(tuned_dt, plot = 'confusion_matrix')


# In[41]:


evaluate_model(tuned_dt)


# # 11.0 Predict on test / hold-out Sample

# In[43]:


predict_model(tuned_dt);


# # 12.0 Finalize Model for Deployment

# In[44]:


final_dt = finalize_model(tuned_dt)


# In[45]:


#Final DecisionTree Classifier model parameters for deployment
print(final_dt)


# In[46]:


predict_model(final_dt);


# # 13.0 Predict on unseen data

# The `predict_model()` function is also used to predict on the unseen dataset. The only difference from section 11 above is that this time we will pass the `data_unseen` parameter. `data_unseen` is the variable created at the beginning of the tutorial and contains 5% (1200 samples) of the original dataset which was never exposed to PyCaret. (see section 5 for explanation)

# In[47]:


unseen_predictions = predict_model(final_dt, data=data_unseen)
unseen_predictions


# The `Label` and `Score` columns are added onto the `data_unseen` set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background. You can also check the metrics on this since you have actual target column `default` available. To do that we will use `pycaret.utils` module. See example below:

# In[48]:


from pycaret.utils import check_metric
check_metric(unseen_predictions['anomaly'], unseen_predictions['anomaly'], metric = 'Accuracy')


# # 14.0 Saving the model

# We have now finished the experiment by finalizing the `tuned_rf` model which is now stored in `final_rf` variable. We have also used the model stored in `final_rf` to predict `data_unseen`. This brings us to the end of our experiment, but one question is still to be asked: What happens when you have more new data to predict? Do you have to go through the entire experiment again? The answer is no, PyCaret's inbuilt function `save_model()` allows you to save the model along with entire transformation pipeline for later use.

# In[49]:


save_model(final_dt, 'Final_Model15022022')


# (TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# # 15.0 Loading the saved model

# To load a saved model at a future date in the same or an alternative environment, we would use PyCaret's `load_model()` function and then easily apply the saved model on new unseen data for prediction.

# In[50]:


saved_final_dt = load_model('Final_Model15022022')


# Once the model is loaded in the environment, you can simply use it to predict on any new data using the same `predict_model()` function. Below we have applied the loaded model to predict the same `data_unseen` that we used in section 13 above.

# In[51]:


new_prediction = predict_model(saved_final_dt, data=data_unseen)


# In[52]:


new_prediction


# Notice that the results of `unseen_predictions` and `new_prediction` are identical.

# In[53]:


from pycaret.utils import check_metric
check_metric(new_prediction['anomaly'], new_prediction['Label'], metric = 'Accuracy')


# # 16.0 Wrap-up / Next Steps?

# This tutorial has covered the entire machine learning pipeline from data ingestion, pre-processing, training the model, hyperparameter tuning, prediction and saving the model for later use. We have completed all of these steps in less than 10 commands which are naturally constructed and very intuitive to remember such as `create_model()`, `tune_model()`, `compare_models()`. Re-creating the entire experiment without PyCaret would have taken well over 100 lines of code in most libraries.
# 
# We have only covered the basics of `pycaret.classification`. In following tutorials we will go deeper into advanced pre-processing, ensembling, generalized stacking and other techniques that allow you to fully customize your machine learning pipeline and are must know for any data scientist.
# 
# See you at the next tutorial. Follow the link to __[Binary Classification Tutorial (CLF102) - Intermediate Level](https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb)__

# In[54]:


pip install Flask


# In[55]:


from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = load_model('Final_Model')
cols = ['Longitude', 'Latitude']

@app.route('/')

def home():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data= data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return render_template("predict.html", pred='Point will be{}'.format(prediction))


# In[ ]:




