import torch
import numpy as np
import sklearn
import sklearn.ensemble
from anchor import anchor_tabular


# (Written by Agnes & Tomas)

# Apply Anchor on a model built on torch.nn.module

def applyAnchor(idx,model,train,test,feature_names,class_names):
    train=train.numpy()
    explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, train)

    idx=idx
    with torch.no_grad():
        test_input = torch.tensor(test[idx], dtype=torch.float32).reshape(1, -1)
        predicted_class = model.predict(test_input)
        print('Prediction: ', explainer.class_names[int(predicted_class[0].item())])  # Access item directly and cast to int
        
        # Convert test_input to a NumPy array before passing it to explain_instance
        test_input_np = test_input.numpy()[0]
        
        # Define a function to predict using the model and return NumPy array
        def _predict_fn(x):
            return model.predict(torch.tensor(x).float()).numpy().astype(int).reshape(-1)
    
            
        # Pass the predict function to explain_instance
        exp = explainer.explain_instance(test_input_np, _predict_fn, threshold=0.95)

    return exp


