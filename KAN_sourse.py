import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import torch
from kan import KAN
from kan.LBFGS import *

from sklearn.base import RegressorMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split


class KAN_es(KAN):
    """
    KAN class with early stopping training. Early sropping was made closly to skl.MLPRegressor .
    """
    def train_es(self, dataset, tol=0.001, n_iter_no_change=10,
                  opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,
                  small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu'):

        ''' Train with early stopping.

        Args:
        -----
        -- Changed --
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], 
                dataset['val_input'], dataset['val_label'], 
                dataset['test_input'], dataset['test_label']
        -- My par-s --
            tol : float
                Delta of validation fit which doesn`t count as fitness improvement. (Tolerence of training).
            n_iter_no_change : int
                Number of iteration with no fit change to early stopping.
        -----
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            device : str
                device   
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['val_loss'], 1D array of validation losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
        '''


        # Early stopping stuff preparation
        no_fit_change_steps = 0
        best_val_rmse = np.inf
        # Remembering first model
        best_model_dict = deepcopy(self.state_dict())

        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)


            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=130)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['val_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_val = dataset['val_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_val = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg(self.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        # Main training loop
        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            val_id= np.random.choice(dataset['val_input'].shape[0], batch_size_val, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            # Early stopping processing stuff
            val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()
            if (val_rmse > best_val_rmse - tol):
                no_fit_change_steps += 1
            else:
                no_fit_change_steps = 0

            if val_rmse < best_val_rmse:
                # Remembering best_val_fit and best_model
                best_val_rmse = val_rmse
                best_model_dict = deepcopy(self.state_dict())


            if _ % log == 0:
                pbar.set_description("trn_ls: %.2e | vl_ls: %.2e | e_stop: %d/%d | tst_ls: %.2e | reg: %.2e " % (
                                                        torch.sqrt(train_loss).cpu().detach().numpy(), 
                                                        val_rmse, 
                                                        no_fit_change_steps,
                                                        n_iter_no_change,
                                                        torch.sqrt(test_loss).cpu().detach().numpy(), 
                                                        reg_.cpu().detach().numpy() ))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['val_loss'].append(val_rmse)
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

            # Checking early stopping criteria
            if no_fit_change_steps==n_iter_no_change:
                print(f'Early stopping criteria raised')
                break
        
        # Load best model
        self.load_state_dict(best_model_dict)
        self(dataset['train_input'])
        val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
        val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()

        return results
    
    
#-- Sci-kit learn KANRegressor --
# ToDo: all done!

class KANRegressor(RegressorMixin, BaseEstimator):
    """Sci-kit learn wrapper for pykan model.
    
    Hierarchical inheritance chain of classes:
    1. pykan.kan --> 
    2. --> KAN_es: pykan.kan with early stopping, made as in skl.neural_network.MLPRegressor -->
    3. --> KANRegressor: pykan.kan wrapped in (RegressorMixin, BaseEstimator) for compatibility with skl interface.
           Uses params, inspired by skl.neural_network.MLPRegressor params logic.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    
    _parameter_constraints = {}
    
    _d_solver_translation = {
        'lbfgs': 'LBFGS',
        'adam': 'Adam'
    }

    def __init__(self, 
                 # kan.__init__ args
                 hidden_layer_sizes=None,
                 grid=3,
                 k=3,
                 seed=1,
                 device='cpu',
                 # kan.train_es args
                 tol=1e-3,
                 n_iter_no_change=10,
                 solver='lbfgs',
                 max_iter=100,
                 learning_rate_init=1.0,
                 validation_fraction=0.1,
                 # additional kan.__init__ args
                 mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0,
                 # additional kan.train_es args
                 log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, stop_grid_update_step=50, batch=-1, small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video'
                 ):
        '''
        Init method. Saving all kan.__init__ and kan.fit args.
        
        Args:
        -----
            -- kan.__init__ parameteres --
            hidden_layer_sizes : list
                the ith element represents the number of neurons in the ith hidden layer.
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True. 
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            
            -- kan.train_es parameteres --
            tol : float
                Delta of validation fit which doesn`t count as fitness improvement. (Tolerence of training).
            n_iter_no_change : int
                Number of iteration with no fit change to early stopping.
            solver : str
                "lbfgs" or "adam"
            max_iter : int
                training steps
            learning_rate_init: float
                learning rate
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            save_fig_freq : int
                save figure every (save_fig_freq) step
        '''
        
        self.hidden_layer_sizes, self.grid, self.k, self.seed, self.device = hidden_layer_sizes, grid, k, seed, device
        self.mult_arity, self.noise_scale, self.scale_base_mu, self.scale_base_sigma, self.base_fun, self.symbolic_enabled, self.affine_trainable, self.grid_eps, self.grid_range, self.sp_trainable, self.sb_trainable, self.save_act, self.sparse_init, self.auto_save, self.first_init, self.ckpt_path, self.state_id, self.round = mult_arity, noise_scale, scale_base_mu, scale_base_sigma, base_fun, symbolic_enabled, affine_trainable, grid_eps, grid_range, sp_trainable, sb_trainable, save_act, sparse_init, auto_save, first_init, ckpt_path, state_id, round
        
        self.tol, self.n_iter_no_change, self.solver, self.max_iter, self.learning_rate_init, self.validation_fraction = tol, n_iter_no_change, solver, max_iter, learning_rate_init, validation_fraction
        self.log, self.lamb, self.lamb_l1, self.lamb_entropy, self.lamb_coef, self.lamb_coefdiff, self.update_grid, self.grid_update_num, self.loss_fn, self.stop_grid_update_step, self.batch, self.small_mag_threshold, self.small_reg_factor, self.metrics, self.sglr_avoid, self.save_fig, self.in_vars, self.out_vars, self.beta, self.save_fig_freq, self.img_folder = log, lamb, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, update_grid, grid_update_num, loss_fn, stop_grid_update_step, batch, small_mag_threshold, small_reg_factor, metrics, sglr_avoid, save_fig, in_vars, out_vars, beta, save_fig_freq, img_folder
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        
        X, y = self._validate_data(X, y, accept_sparse=True)
        n_features = X.shape[1]
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))
            
        if not hasattr(self, "is_fitted_"): self.is_fitted_ = False
        
        if not self.is_fitted_:
            self.n_outputs_ = y.shape[1]
        
            self.width = [n_features] + self.hidden_layer_sizes + [self.n_outputs_]
            self.kan = KAN_es(width=self.width, grid=self.grid, k=self.k, seed=self.seed, device=self.device,
                              mult_arity=self.mult_arity, noise_scale=self.noise_scale, scale_base_mu=self.scale_base_mu, scale_base_sigma=self.scale_base_sigma, base_fun=self.base_fun, symbolic_enabled=self.symbolic_enabled, affine_trainable=self.affine_trainable, grid_eps=self.grid_eps, grid_range=self.grid_range, sp_trainable=self.sp_trainable, sb_trainable=self.sb_trainable, save_act=self.save_act, sparse_init=self.sparse_init, auto_save=self.auto_save, first_init=self.first_init, ckpt_path=self.ckpt_path, state_id=self.state_id, round=self.round)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                        test_size=self.validation_fraction,
                                                        random_state=self.seed)
                    
        kan_dataset = {'train_input': torch.tensor(np.array(X_train), dtype=torch.float),
                       'train_label': torch.tensor(np.array(y_train), dtype=torch.float),
                       'val_input': torch.tensor(np.array(X_val), dtype=torch.float),
                       'val_label': torch.tensor(np.array(y_val), dtype=torch.float),
                       # Test data is required in KAN_es.fit_es . So here we should ignore test datasets.
                       'test_input': torch.tensor(np.array([X[0], X[1]]), dtype=torch.float),
                       'test_label': torch.tensor(np.array([y[0], y[1]]), dtype=torch.float)}
        
        self.kan.train_es(kan_dataset, 
                          tol=self.tol, 
                          n_iter_no_change=self.n_iter_no_change,
                          opt=self._d_solver_translation[self.solver], 
                          lr = self.learning_rate_init,
                          steps=self.max_iter,
                          log=self.log, lamb=self.lamb, lamb_l1=self.lamb_l1, lamb_entropy=self.lamb_entropy, lamb_coef=self.lamb_coef, lamb_coefdiff=self.lamb_coefdiff, update_grid=self.update_grid, grid_update_num=self.grid_update_num, loss_fn=self.loss_fn, stop_grid_update_step=self.stop_grid_update_step, batch=self.batch, small_mag_threshold=self.small_mag_threshold, small_reg_factor=self.small_reg_factor, metrics=self.metrics, sglr_avoid=self.sglr_avoid, save_fig=self.save_fig, in_vars=self.in_vars, out_vars=self.out_vars, beta=self.beta, save_fig_freq=self.save_fig_freq, img_folder=self.img_folder, device=self.device
                          )
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        
        x = torch.tensor(np.array(X), dtype=torch.float)
        pred = self.kan.forward(x).detach().numpy()
        
        return pred