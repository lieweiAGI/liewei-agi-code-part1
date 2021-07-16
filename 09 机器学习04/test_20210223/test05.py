#弥补缺失数据
from sklearn.impute import SimpleImputer
import numpy as np

data = np.array([[np.nan,2],[6,np.nan],[7,6],[8,6]])
print(data)
imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
y_imp = imp.fit_transform(data)
print(y_imp)