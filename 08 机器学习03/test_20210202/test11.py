from sklearn import datasets,preprocessing,linear_model,neighbors,svm
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
import joblib

#加载数据
iris = datasets.load_iris()
x,y = iris.data,iris.target

#划分数据
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#创建模型
# clf = neighbors.KNeighborsClassifier(n_neighbors=6)
# clf = linear_model.SGDClassifier()
# clf = svm.SVC()
clf = linear_model.LogisticRegression()

#模型拟合
clf.fit(x_train,y_train)
#预测
y_pred = clf.predict(x_test)
#评估
print(accuracy_score(y_test,y_pred))
#保存模型
joblib.dump(clf,"params/SGD_clf.pkl")