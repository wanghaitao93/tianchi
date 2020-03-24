# 数据分析


## 目标
- EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。

- 当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。



## 工具

- 载入各种数据科学以及可视化库:

	数据科学库 pandas、numpy、scipy；
	
	可视化库 matplotlib、seabon；
	
- 载入数据：

	载入训练集和测试集；
	
	简略观察数据(head()+shape)；


- 数据总览:

	通过describe()来熟悉数据的相关统计量
	
	通过info()来熟悉数据类型

- 判断数据缺失和异常

	查看每列的存在nan情况

	异常值检测


- 了解预测值的分布

	总体分布概况（无界约翰逊分布等）

	查看skewness and kurtosis

	查看预测值的具体频数

- 特征分为类别特征和数字特征，并对类别特征查看unique分布

- 数字特征分析

	相关性分析
	
	查看几个特征得 偏度和峰值
	
	每个数字特征得分布可视化
	
	数字特征相互之间的关系可视化
	
	多变量互相回归关系可视化
	
- 类型特征分析
	
	unique分布

	类别特征箱形图可视化

	类别特征的小提琴图可视化

	类别特征的柱形图可视化类别
 
 	特征的每个类别频数可视化(count_plot)

- 用pandas_profiling生成数据报告


## 实践

### 常用包

```
## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore') inline
%matplotlib 

## 模型预测
from sklearn import linear_model 
from sklearn import preprocessing 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA 
import lightgbm as lgb 
import xgboost as xgb

## 参数搜索和评价
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

### 数据查看

```
查看几个数据
df.head()
df.info()
df.columns
统计信息
df.describe()
提取数值类型与特征列名
df.select_dtypes(exclude='object').columns
df.select_dtypes(include='object').columns
查看每列缺失值情况
df.isnull().sum()
缺失值可视化
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
import missingno as msno
msno.matrix(Test_data.sample(250))
填补缺失值
df.fillna(-1)
可能存在的其他类型的缺失值替换
df['cloumns_name'].replace('-', np.nan, inplace=True)
值统计
df['cloumns_name'].value_counts()
预测值分布
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```