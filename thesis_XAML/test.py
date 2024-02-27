
import sys
sys.path.append('/home/agnes/desktop/flib/gnn')
from main import * 
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn

vectorizer = TfidfVectorizer(min_df=10)


model=train_logistic_regressor()

traindata = AmlsimDataset(node_file='data/simtest/swedbank/train/nodes.csv', edge_file='data/simtest/swedbank/train/edges.csv', node_features=True, node_labels=True).get_data()
testdata = AmlsimDataset(node_file='data/simtest/swedbank/test/nodes.csv', edge_file='data/simtest/swedbank/test/edges.csv', node_features=True, node_labels=True).get_data()

testdata_x_numpy = testdata.x.numpy()  # Convert to NumPy array
traindata_x_numpy = traindata.x.numpy()  # Convert to NumPy array
print(testdata_x_numpy.shape)
#model.eval()

#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

#masker = shap.maskers.Independent(data=testdata_x_numpy, max_samples=10)
#explainer = shap.Explainer(model, masker)


print(testdata.y[0:10])




#shap_values = explainer.shap_values(testdata_x_numpy)


