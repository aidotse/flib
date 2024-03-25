import torch
from torch.autograd import Variable
import numpy

import lime
import shap
import shap

def LIME_explanation(node_to_explain, num_features, class_prob_fn, testdata, feature_names, target_names):
    # Wrap the forward function of the model
    # Note: Make sure that class_prob_fn returns class probabilites for both classes
    wrapped_class_prob_fn = lambda x: class_prob_fn(torch.from_numpy(x).float()).numpy()

    # --- Insert extraction of subgraph here for reducing the runtime by 50% ---

    # Create explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(testdata.x.to('cpu').numpy(),
                                                    feature_names=feature_names,
                                                    class_names=target_names)

    exp_LIME = explainer.explain_instance(testdata.x[node_to_explain].to('cpu').numpy(), wrapped_class_prob_fn, num_features=num_features)
    
    return exp_LIME

def SHAP_explanation(node_to_explain, class_prob_fn, backgrounddata, explaindata, feature_names, K):
    # Wrap the forward function of the model
    # Note: Make sure that class_prob_fn returns class probabilites for both classes
    is_sar_class_prob_fn = lambda x: class_prob_fn( Variable( torch.from_numpy(x) ) ).detach().numpy()[:,1].reshape(-1,1)

    # --- Insert extraction of subgraph here for reducing the runtime by 50% ---

    # ...fattar inte riktigt vad NFV ska ha för shape egentligen, men det här verkar funka iaf.
    NFV_to_explain = explaindata.x[node_to_explain].to('cpu').numpy().squeeze() 
    print(NFV_to_explain)
    explainer = shap.KernelExplainer(is_sar_class_prob_fn, shap.sample(backgrounddata.x.to('cpu').numpy(), K))
    shap_values = explainer.shap_values(NFV_to_explain)
    exp_SHAP = shap.Explanation(shap_values[0], explainer.expected_value, data = NFV_to_explain, feature_names = feature_names)
    
    return exp_SHAP, shap_values

def Counterfactual_explanation(node_to_explain):
    print('Not implemented yet')