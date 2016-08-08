import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import linear_model, datasets, cross_validation
import matplotlib.pyplot as plt

#4.1
dataset = []
f = open('./fraud_data_3.csv', 'rU')
try:
    reader = csv.reader(f,delimiter=',')
    next(reader, None)
    for row in reader:
        dataset.append(row)
finally:
    f.close()

#4.2
target = np.array([x[0] for x in dataset])
data = np.array([x[1:] for x in dataset])
# Amount, Country, TimeOfTransaction, BusinessType, NumberOfTransactionsAtThisShop, DayOfWeek
categorical_mask = [False,True,True,True,False,True]
enc = LabelEncoder()

for i in range(0,data.shape[1]):
    if(categorical_mask[i]):
        label_encoder = enc.fit(data[:,i])
        print "Categorical classes:", label_encoder.classes_
        integer_classes = label_encoder.transform(label_encoder.classes_)
        print "Integer classes:", integer_classes        
        t = label_encoder.transform(data[:, i])
        data[:, i] = t

#4.3:
mask = np.ones(data.shape, dtype=bool)

for i in range(0,data.shape[1]):
    if(categorical_mask[i]):
        mask[:,i]=False

data_non_categoricals = data[:, np.all(mask, axis=0)] #keep only the true, non categoricals
data_categoricals = data[:,~np.all(mask,axis=0)]

hotenc = OneHotEncoder()
hot_encoder = hotenc.fit(data_categoricals)
encoded_hot = hot_encoder.transform(data_categoricals)

#4.4:
new_data=data_non_categoricals
new_data=new_data.astype(np.float)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_data, target, test_size=0.4, random_state=0,dtype=float)

logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)						
log_output = logreg.predict_log_proba(X_test)				

print("Odds: "+ str(np.exp(logreg.coef_)))
print("Odds intercept" + str(np.exp(logreg.intercept_)))
print("Likelihood Intercept:" + str(np.exp(logreg.intercept_)/(1+np.exp(logreg.intercept_))))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.setp((ax1,ax2),xticks=[])

ax1.scatter(range(0,len(log_output[:,1]),1),log_output[:,1],s=100,label='Log Prob.',color='Blue',alpha=0.5)
ax1.scatter(range(0,len(y_test),1),y_test,label='Labels',s=250,color='Green',alpha=0.5)
ax1.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), ncol=2, loc=3, mode="expand", borderaxespad=0.)
ax1.set_xlabel('Test Instances')
ax1.set_ylabel('Binary Ground Truth Labels /  Model Log. Prob.')

prob_output = [np.exp(x) for x in log_output[:,1]]
ax2.scatter(range(0,len(prob_output),1),prob_output,s=100,label='Prob.', color='Blue',alpha=0.5)
ax2.scatter(range(0,len(y_test),1),y_test,label='Labels',s=250,color='Green',alpha=0.5)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), ncol=2, loc=3, mode="expand", borderaxespad=0.)
ax2.set_xlabel('Test Instances')
ax2.set_ylabel('Binary Ground Truth Labels / Model Prob.')

plt.show()

#4.5:
new_data = np.append(data_non_categoricals,encoded_hot.todense(),1)
new_data=new_data.astype(np.float)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_data, target, test_size=0.4, random_state=0,dtype=float)

logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)
log_output = logreg.predict_log_proba(X_test)

print("Odds: "+ str(np.exp(logreg.coef_)))
print("Odds intercept" + str(np.exp(logreg.intercept_)))
print("Likelihood Intercept:" + str(np.exp(logreg.intercept_)/(1+np.exp(logreg.intercept_))))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.setp((ax1,ax2),xticks=[])

ax1.scatter(range(0,len(log_output[:,1]),1),log_output[:,1],s=100,label='Log Prob.',color='Blue',alpha=0.5)
ax1.scatter(range(0,len(y_test),1),y_test,label='Labels',s=250,color='Green',alpha=0.5)
ax1.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), ncol=2, loc=3, mode="expand", borderaxespad=0.)
ax1.set_xlabel('Test Instances')
ax1.set_ylabel('Binary Ground Truth Labels /  Model Log. Prob.')

prob_output = [np.exp(x) for x in log_output[:,1]]
ax2.scatter(range(0,len(prob_output),1),prob_output,s=100,label='Prob.', color='Blue',alpha=0.5)
ax2.scatter(range(0,len(y_test),1),y_test,label='Labels',s=250,color='Green',alpha=0.5)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), ncol=2, loc=3, mode="expand", borderaxespad=0.)
ax2.set_xlabel('Test Instances')
ax2.set_ylabel('Binary Ground Truth Labels / Model Prob.')

plt.show()


