#%%
import numpy as np
import pandas as pd

from uGUIDE.config_utils import create_config_uGUIDE, save_config_uGUIDE
from uGUIDE.data_utils import preprocess_data, postprocess_SM
from uGUIDE.training import run_training
from uGUIDE.estimation import estimate_microstructure

#%%
theta_train = pd.read_csv('simulations_SM_train_10000__f_Da_ODI_u0_u1.csv',
                          header=None).values
x_train = pd.read_csv(f'simulations_SM_train_10000__S_snr_50.csv',
                      header=None).values

bvals = np.loadtxt('bvals.bval')
theta_train, x_train = preprocess_data(theta_train,
                                       x_train,
                                       bvals,
                                       normalize=True)

#%%
eps = 1e-3
prior = {
    'f': [0.0, 1.0],
    'Da': [0.1, 3.0],
    'ODI': [0.03, 0.95],
    'u0': [0.0, 1.0],
    'u1': [0.0, 1.0]
}
prior_postprocessing = {
    'f': [0.0, 1.0],
    'Da': [0.1, 3.0],
    'ODI': [0.03, 0.95],
    'De_par': [0.1, 3.0],
    'De_perp': [0.1, 3.0]
}
theta_train[:, 0] = np.clip(theta_train[:, 0], prior['f'][0] + eps,
                            prior['f'][1] - eps)
theta_train[:, 1] = np.clip(theta_train[:, 1], prior['Da'][0] + eps,
                            prior['Da'][1] - eps)
theta_train[:, 2] = np.clip(theta_train[:, 2], prior['ODI'][0] + eps,
                            prior['ODI'][1] - eps)
theta_train[:, 3] = np.clip(theta_train[:, 3], prior['u0'][0] + eps,
                            prior['u0'][1] - eps)
theta_train[:, 4] = np.clip(theta_train[:, 4], prior['u1'][0] + eps,
                            prior['u1'][1] - eps)

#%%
config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                              size_x=x_train.shape[1],
                              prior=prior,
                              prior_postprocessing=prior_postprocessing,
                              use_MLP=True,
                              nf_features=12,
                              learning_rate=5e-4,
                              scheduler_patience=15,
                              max_epochs=1000,
                              random_seed=1234)
save_config_uGUIDE(config, savefile='config_postprocessing.pkl')
print(f'Device used for computations: {config["device"]}')

#%%
run_training(theta_train,
             x_train,
             config=config,
             plot_loss=True,
             load_state=False)

#%%
theta_test = pd.read_csv('simulations_SM_test_1000__f_Da_ODI_u0_u1.csv',
                         header=None).values
x_test = pd.read_csv(f'simulations_SM_test_1000__S_snr_50.csv',
                     header=None).values
theta_test, x_test = preprocess_data(theta_test, x_test, bvals, normalize=True)

#%%
# Without postprocessing
_ = estimate_microstructure(x_test[:5, :],
                            config,
                            verbose=True,
                            plot=True,
                            theta_gt=theta_test[:5, :])

# %%
# With postprocessing
# Convert u0 and u1 to De_par and De_perp and plot results
_ = estimate_microstructure(x_test[:5, :],
                            config,
                            postprocessing=postprocess_SM,
                            verbose=True,
                            plot=True)

# %%
# import torch
# import pyro.distributions as dist
# from uGUIDE.embedded_net import get_embedded_net
# from uGUIDE.density_estimator import get_nf

# # Take two noticeably different test MRI signals
# x1 = torch.Tensor(x_test[0:1, :]).to(config['device'])
# x2 = torch.Tensor(x_test[12:13, :]).to(config['device'])

# load_state = True
# device = config['device']

# embedded_net = get_embedded_net(
#         input_dim=config['size_x'],
#         output_dim=config['nf_features'],
#         layer_1_dim=config['hidden_layers'][0],
#         layer_2_dim=config['hidden_layers'][1],
#         pretrained_state=config['folderpath'] /
#         config['embedder_state_dict_file'] if load_state else None,
#         use_MLP=config['use_MLP'], device=config['device']).to(config['device'])
# nf = get_nf(input_dim=config['size_theta'],
#                 nf_features=config['nf_features'],
#                 n_flows=config['n_flows'],
#                 pretrained_state=config['folderpath'] /
#                 config['nf_state_dict_file'],
#                 device='cuda').to('cuda')
# embedded_net.eval()
# nf.eval()
# B = x1.shape[0]  # Actual batch size (last batch might be smaller)
# D = config['size_theta']

# with torch.no_grad():
#     emb1 = embedded_net(x1.to(config['device']))
#     emb2 = embedded_net(x2.to(config['device']))

#     # Print embedding outputs — they MUST be different!
#     print("Embedding 1:", emb1)
#     print("Embedding 2:", emb2)
#     print("Difference norm:", torch.norm(emb1 - emb2).item())

#     base_dist = dist.Normal(
#                 torch.zeros((B, D), device=device),
#                 torch.ones((B, D), device=device),
#             ).to_event(1)
#     transformed_dist = dist.ConditionalTransformedDistribution(
#                     base_dist, nf)
#     # Sample conditional distribution
#     dist1 = transformed_dist.condition(emb1)
#     dist2 = transformed_dist.condition(emb2)

#     s1 = dist1.sample((1000,))
#     s2 = dist2.sample((1000,))

#     print("Sample 1 Mean:", s1.mean(dim=0))
#     print("Sample 2 Mean:", s2.mean(dim=0))
