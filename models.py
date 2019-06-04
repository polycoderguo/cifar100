from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

class modeltrainer:
	def __init__(self, x_train, x_test, y_train, y_test):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
	def train_GradientBoost():
		clf = GradientBoostingClassifier()
		clf.fit(self.x_train_, self.y_train)
		score = clf.score(self.x_test, y_test)
        	#print(score)
		return score
	def train_XGBoost()



