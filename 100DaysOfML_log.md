## Day 0 : July 06 , 2018
 
**Today's Progress** : I have setup all the things I needed to complete this challenge and also completed chosing the projects I will work on .
  
**Thoughts** : Hope this will be exiciting ,will help me in learning Machine Learning in a more effective way .

## Day 1 : July 07 , 2018
 
**Today's Progress** : Warm Up: Predict Blood Donations
  
**Thoughts** : I completed EDA on Blood Donations data. 
**Project Description:**   [Predicting Blood Donations](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/)

'''
f = plt.figure(figsize=(16,16))
ax1 = f.add_subplot(4,1,1)
ax2 = f.add_subplot(4,1,2)
ax3 = f.add_subplot(4,1,3)
ax4 = f.add_subplot(4,1,4)

ax1.set_title('log Months since Last Donation')
sns.kdeplot(train_data['log Months since Last Donation'], shade=True, cut=0, label='train_data',ax=ax1)
sns.kdeplot(test_data['log Months since Last Donation'], shade=True, cut=0, label='test_data',ax=ax1)

ax2.set_title('log Number of Donations')
sns.kdeplot(train_data['log Number of Donations'], shade=True, cut=0, label='train_data',ax=ax2)
sns.kdeplot(test_data['log Number of Donations'], shade=True, cut=0, label='test_data',ax=ax2)

ax3.set_title('log Total Volume Donated (c.c.)')
sns.kdeplot(train_data['log Total Volume Donated (c.c.)'], shade=True, cut=0, label='train_data',ax=ax3)
sns.kdeplot(test_data['log Total Volume Donated (c.c.)'], shade=True, cut=0, label='test_data',ax=ax3)

ax4.set_title('log Months since First Donation')
sns.kdeplot(train_data['log Months since First Donation'], shade=True, cut=0, label='train_data',ax=ax4)
sns.kdeplot(test_data['log Months since First Donation'], shade=True, cut=0, label='test_data',ax=ax4)

plt.show()
'''


