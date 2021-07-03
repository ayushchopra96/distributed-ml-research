##################################
# Client
# : actual training model that 
# learns a sequence of tasks.
# For each communication round,
# it receives parameters from server
# and train on given task. It sends 
# its updates back to server after 
# train step is done.
##################################
__author__ = "Wonyong Jeong, Jaehong Yoon"
__credits__ = ["Wonyong Jeong", "Jaehong Yoon"]
__email__ = "wyjeong@kaist.ac.kr, jaehong.yoon@kaist.ac.kr"
##################################

import pdb
import math
import random
import threading
import tensorflow as tf 
import tensorflow.keras as tf_keras
import tensorflow.keras.models as tf_models
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.activations as tf_activations
from .layers import *
from utils.fileio import *
from data.data_loader import DataLoader

lock = threading.Lock()

class Client:
    
    def __init__(self, client_id, opt, initial_weights):
        self.opt = opt
        self.client_id = client_id
        self.curr_task = -1
        self.done = False
        self.early_stop = False
        self.task_names = []
        self.local_models = []
        self.x_test_list = []
        self.y_test_list = []
        self.x_valid = []
        self.y_valid = []
        self.performance_task = {}
        self.data_loader = DataLoader(self.opt, client_id)
        self.initializer = tf.keras.initializers.VarianceScaling()
        self.variables = {'mask':{}, 'bias':{}, 'adaptive':{}, 'from_kb': {}, 'atten': {}}
        self.metrics = {
            'valid_lss': tf.keras.metrics.Mean(name='valid_lss'),
            'train_lss': tf.keras.metrics.Mean(name='train_lss'),
            'test_lss' : tf.keras.metrics.Mean(name='test_lss'),
            'valid_acc': tf.keras.metrics.CategoricalAccuracy(name='valid_acc'),
            'train_acc': tf.keras.metrics.CategoricalAccuracy(name='train_acc'),
            'test_acc' : tf.keras.metrics.CategoricalAccuracy(name='test_acc')
        }
        self.heads = {}
        self.capacity = []
        self.communication = []
        self.init_models(initial_weights)

    def get_optimizer(self, curr_lr):
        return tf.keras.optimizers.SGD(learning_rate=curr_lr, momentum=self.opt.momentum)

    def init_models(self, initial_weights):
        self.local_model_body = self.build_model_body(initial_weights)
        for tid in range(self.opt.num_tasks):
            self.add_head(tid)

    def build_model_body(self, initial_weights):
        tid = 0 
        apc_lid = 0
        lock.acquire()
        model = tf_models.Sequential()
        if self.opt.base_network == 'alexnet-like':
            self.apc_layers = {}
            self.shapes = [
                (4, 4, 3, 64),
                (3, 3, 64, 128),
                (2, 2, 128, 256),
                (4096, 2048),
                (2048, 2048),
            ]
            if 'shared' not in self.variables:
                self.variables['shared'] = [tf.Variable(initial_weights[i], trainable=True, 
                                name='layer_{}/sw'.format(i)) for i in range(len(self.shapes))]
            self.conv_layers = [0, 1, 2]
            for lid in self.conv_layers:
                self.apc_layers[apc_lid] = DecomposedConv(
                    input_shape = (32,32,3),
                    name        = 'layer_{}'.format(lid),
                    filters     = self.shapes[lid][-1], 
                    kernel_size = (self.shapes[lid][0], self.shapes[lid][1]), 
                    strides     = (1, 1), 
                    padding     = 'same', 
                    activation  = 'relu',
                    lambda_l1   = self.opt.lambda_l1,
                    lambda_mask = self.opt.lambda_mask,
                    shared      = self.variables['shared'][lid],
                    adaptive    = self.create_variable('adaptive', lid, tid),
                    from_kb     = self.create_variable('from_kb', lid, tid),
                    atten       = self.create_variable('atten', lid, tid),
                    bias        = self.create_variable('bias', lid, tid), use_bias=True,
                    mask        = self.generate_mask(self.create_variable('mask', lid, tid)))
                model.add(self.apc_layers[apc_lid])
                apc_lid += 1
                if lid<2:
                    model.add(tf_layers.Dropout(0.2))
                else:
                    model.add(tf_layers.Dropout(0.5))
                model.add(tf_layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf_layers.Flatten())
            self.fc_layers = [3, 4]
            for lid in self.fc_layers:
                self.apc_layers[apc_lid] = DecomposedDense( 
                    name        = 'layer_{}'.format(lid),
                    units       = self.shapes[lid][-1], 
                    input_dim   = self.shapes[lid][0],
                    lambda_l1   = self.opt.lambda_l1,
                    lambda_mask = self.opt.lambda_mask,
                    shared      = self.variables['shared'][lid],
                    adaptive    = self.create_variable('adaptive', lid, tid),
                    from_kb     = self.create_variable('from_kb', lid, tid),
                    atten       = self.create_variable('atten', lid, tid),
                    bias        = self.create_variable('bias', lid, tid), use_bias=True,
                    mask        = self.generate_mask(self.create_variable('mask', lid, tid)))
                model.add(self.apc_layers[apc_lid])
                apc_lid += 1
                model.add(tf_layers.Dropout(0.5))
        lock.release()
        return model
    
    def add_head(self, tid):
        self.heads[tid] = tf_layers.Dense(self.opt.num_classes, activation='softmax')
        body_out = self.local_model_body.output
        head_out = self.heads[tid](body_out)
        model = tf.keras.Model(inputs=self.local_model_body.input, outputs=head_out)
        self.local_models.append(model) # multiheaded model 

    def create_variable(self, var_type, lid, tid):
        if tid not in self.variables[var_type]:
            self.variables[var_type][tid] = {}
        if var_type == 'adaptive':
            trainable = True
            init_value = self.variables['shared'][lid].numpy()/3
        elif var_type == 'from_kb':
            trainable = False
            _shape = list(self.shapes[lid])
            _shape.append(int(self.opt.num_clients))
            _shape = tuple(_shape)
            init_value = np.zeros(_shape, dtype=np.float32)
        elif var_type == 'atten':
            trainable = True if tid > 0 else False
            init_value = self.initializer((int(self.opt.num_clients), ))
        else:
            trainable = True
            init_value = self.initializer((self.shapes[lid][-1], ))
        var = tf.Variable(init_value, trainable=trainable, name='layer_{}/task_{}/{}'.format(lid, tid, var_type))
        self.variables[var_type][tid][lid] = var
        return var

    def get_variable(self, var_type, lid, tid=None):
        if var_type == 'shared':
            return self.variables[var_type][lid]
        else:
            if tid not in self.variables[var_type]:
                return self.create_variable(var_type, lid, tid)
            elif lid not in self.variables[var_type][tid]:
                return self.create_variable(var_type, lid, tid)
            else:
                return self.variables[var_type][tid][lid]
    
    def generate_mask(self, mask):
        return tf_activations.sigmoid(mask)

    def init_new_task(self):
        self.curr_task += 1
        self.round_cnt = 0
        # get new task
        data = self.data_loader.get_task(self.curr_task)
        self.task_names.append(data['name'])
        # train data
        train = data['train']
        self.x_train = np.array([tup[0] for tup in train])
        self.y_train = np.array([tup[1] for tup in train])
        # test data
        test = data['test']
        self.x_test_list.append(np.array([tup[0] for tup in test]))
        self.y_test_list.append(np.array([tup[1] for tup in test]))
        # valid data
        valid = data['valid']
        self.x_valid = np.array([tup[0] for tup in valid])
        self.y_valid = np.array([tup[1] for tup in valid])
        # init optimizer $ learning rate
        self.early_stop = False
        self.lowest_lss = np.inf
        self.curr_lr = self.opt.lr
        self.curr_lr_patience = self.opt.lr_patience
        self.optimizer = self.get_optimizer(self.opt.lr)
        # switch model
        self.switch_model_params(self.curr_task)
        # restore prev theta
        self.recover_prev_theta()
        # update trainable variables
        prev_variables = ['mask', 'bias', 'adaptive', 'atten'] 
        self.trainable_variables = [sw for sw in self.variables['shared']]
        for tid in range(self.curr_task+1):
            for lid in range(len(self.shapes)):
                for pvar in prev_variables:
                    if pvar == 'bias' and tid < self.curr_task:
                        continue
                    elif 'atten' in pvar and (tid==0 or tid!=self.curr_task):
                        continue
                    self.trainable_variables.append(self.get_variable(pvar, lid, tid))
        # print('self.trainable_variables:', [v.name for v in self.trainable_variables])
        # print('-----------------------------------------------------------------')
        console_log('[client:{}] appended new task (task_{}: {})'
            .format(self.client_id, self.curr_task, self.task_names[self.curr_task]))

    def switch_model_params(self, tid):
        for lid, lay in self.apc_layers.items():
            lay.aw = self.get_variable('adaptive', lid, tid)
            lay.atten = self.get_variable('atten', lid, tid)
            lay.bias = self.get_variable('bias', lid, tid)
            lay.mask = self.generate_mask(self.get_variable('mask', lid, tid))

    def recover_prev_theta(self): 
        self.prev_theta = {}
        for i in range(len(self.shapes)):
            self.prev_theta[i] = {}
            sw = self.get_variable(var_type='shared', lid=i)
            for j in range(self.curr_task):
                prev_aw = self.get_variable(var_type='adaptive', lid=i, tid=j)
                prev_mask = self.get_variable(var_type='mask', lid=i, tid=j) 
                prev_mask_sig = self.generate_mask(prev_mask)
                #################################################
                prev_theta = sw * prev_mask_sig + prev_aw
                self.prev_theta[i][j] = prev_theta.numpy()
                #################################################

    def get_model_for_task(self, tid):
        self.switch_model_params(tid)
        return self.local_models[tid]

    def train_one_round(self, curr_round, global_weights=None):
        if self.curr_task < 0:
            self.init_new_task()
        else:
            is_last_task = (self.curr_task==self.opt.num_tasks-1)
            is_last_round = (self.round_cnt%self.opt.num_rounds==0 and self.round_cnt!=0)
            is_last = is_last_task and is_last_round
            if is_last_round or self.early_stop:
                if is_last_task:
                    if self.early_stop:
                        self.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
        self.round_cnt += 1
        self.curr_round = curr_round
        for epoch in range(self.opt.num_epochs):
            self.curr_epoch = epoch
            self.train_size = 0
            target_model = self.get_model_for_task(self.curr_task)
            for i in range(0, len(self.x_train), self.opt.batch_size): # train
                x_batch = self.x_train[i:i+self.opt.batch_size]
                y_batch = self.y_train[i:i+self.opt.batch_size]
                y_pred, loss = self.train_step(target_model, x_batch, y_batch)
                self.train_size += len(x_batch)
            # validation & adaptive lr decay
            vlss, vacc = self.validate()
            self.adaptive_lr_decay(vlss)
            # Epoch-level evalation
            self.evaluate()
            if self.early_stop:
                break
        self.calculate_capacity()
        self.log_performance()
    
    def adaptive_lr_decay(self, vlss):
        if vlss<self.lowest_lss:
            self.lowest_lss = vlss
            self.curr_lr_patience = self.opt.lr_patience
        else:
            self.curr_lr_patience-=1
            if self.curr_lr_patience<=0:
                self.curr_lr/=self.opt.lr_factor
                console_log('[client:%d] task:%d, round:%d (cnt:%d), drop lr => %.10f' 
                        %(self.client_id, self.curr_task, self.curr_round, self.round_cnt, self.curr_lr))
                if self.curr_lr<self.opt.lr_min:
                    console_log('[client:%d] task:%d, round:%d (cnt:%d), early stopped, reached minium lr (%.10f)'
                         %(self.client_id, self.curr_task, self.curr_round, self.round_cnt, self.curr_lr))
                    self.early_stop = True
                self.curr_lr_patience = self.opt.lr_patience
                self.optimizer = self.get_optimizer(self.curr_lr)

    def validate(self):
        for i in range(0, len(self.x_valid), self.opt.batch_size):
            x_batch = self.x_valid[i:i+self.opt.batch_size]
            y_batch = self.y_valid[i:i+self.opt.batch_size]
            y_pred, loss = self.test_step(self.local_models[self.curr_task], x_batch, y_batch)
            self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)
        return self.measure_performance('valid_lss', 'valid_acc')

    def evaluate(self): 
        for tid in range(len(self.x_test_list)):
            model = self.get_model_for_task(tid)
            lss, acc = self._evaluate(model, self.x_test_list[tid], self.y_test_list[tid])
            if not tid in self.performance_task:
                self.performance_task[tid] = {
                    'epoch': [],
                    'round': [],
                }
            self.performance_task[tid]['epoch'].append(acc)
            if self.curr_epoch == self.opt.num_epochs-1:
                self.performance_task[tid]['round'].append(acc)
            console_log('[client:%d] round:%d (cnt:%d), epoch:%d, lss:%.3f, acc:%.3f (task_%d:%s, #train:%d, #test:%d)' 
                 %(self.client_id, self.curr_round, self.round_cnt, self.curr_epoch, lss, acc, tid,
                    self.task_names[tid], self.train_size, len(self.x_test_list[tid])))

    def _evaluate(self, model, x_test, y_test):
        for i in range(0, len(x_test), self.opt.batch_size):
            x_batch = x_test[i:i+self.opt.batch_size]
            y_batch = y_test[i:i+self.opt.batch_size]
            y_pred, loss = self.test_step(model, x_batch, y_batch)
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        return self.measure_performance('test_lss', 'test_acc')

    def train_step(self, model, x, y):
        tf.keras.backend.set_learning_phase(1)
        with tf.GradientTape() as tape:
            y_pred = model(x) 
            loss = self.loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return y_pred, loss
        
    def test_step(self, model, x, y):
        tf.keras.backend.set_learning_phase(0)
        y_pred = model(x)
        loss = self.loss(y, y_pred)
        return y_pred, loss

    def loss(self, y_true, y_pred):
        weight_decay, sparseness, approx_loss = 0, 0, 0
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        for i in range(len(self.shapes)):
            sw = self.get_variable(var_type='shared', lid=i)
            aw = self.get_variable(var_type='adaptive', lid=i, tid=self.curr_task)
            mask = self.get_variable(var_type='mask', lid=i, tid=self.curr_task)
            g_mask = self.generate_mask(mask)
            weight_decay += self.opt.wd * tf.nn.l2_loss(aw)
            weight_decay += self.opt.wd * tf.nn.l2_loss(mask)
            sparseness += self.opt.lambda_l1 * tf.reduce_sum(tf.abs(aw))
            sparseness += self.opt.lambda_mask * tf.reduce_sum(tf.abs(mask))
            if self.curr_task == 0:
                weight_decay += self.opt.wd * tf.nn.l2_loss(sw)
            else:
                for j in range(self.curr_task):
                    prev_aw = self.get_variable(var_type='adaptive', lid=i, tid=j)
                    prev_mask = self.get_variable(var_type='mask', lid=i, tid=j) 
                    g_prev_mask = self.generate_mask(prev_mask)
                    #################################################
                    restored = sw * g_prev_mask + prev_aw
                    a_l2 = tf.nn.l2_loss(restored-self.prev_theta[i][j])
                    approx_loss += self.opt.lambda_l2 * a_l2
                    #################################################
                    sparseness += self.opt.lambda_l1 * tf.reduce_sum(tf.abs(prev_aw))
        loss += weight_decay + sparseness + approx_loss
        return loss
  
    def add_performance(self, lss_name, acc_name, loss, y_true, y_pred,):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)

    def measure_performance(self, lss_name, acc_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        return lss, acc

    def stop(self):
        console_log('[client:{}] finished learning all tasks'.format(self.client_id))
        console_log('[client:{}] done.'.format(self.client_id))
        self.done = True

    def get_weights(self):
        if self.opt.sparse_comm:
            hard_threshold = []
            sw_pruned = []
            masks = self.variables['mask'][self.curr_task]
            for lid, sw in enumerate(self.variables['shared']):
                mask = masks[lid]
                m_sorted = tf.sort(tf.keras.backend.flatten(tf.abs(mask)))
                thres = m_sorted[math.floor(len(m_sorted)*(self.opt.client_sparsity))]
                m_bianary = tf.cast(tf.greater(tf.abs(mask), thres), tf.float32).numpy().tolist()
                hard_threshold.append(m_bianary)
                sw_pruned.append(sw.numpy()*m_bianary)
            self.calculate_comm_costs(sw_pruned)
            return sw_pruned, hard_threshold
        else:
            return [sw.numpy() for sw in self.variables['shared']], None

    def get_adapts(self):
        params =  [self.l1_pruning(adp, self.opt.lambda_l1).numpy() for adp in self.variables['adaptive'][self.curr_task]]
        return params

    def get_both(self):
        gw=self.get_weights()
        ga=self.get_adapts()
        return np.array([gw, ga])

    def set_both(self, new_weights, update_ta=False):
        self.set_weights(new_weights[0])
        if update_ta:
            self.set_adapts(new_weights[1])

    def set_weights(self, new_weights):
        for i, w in enumerate(new_weights):
            sw = self.get_variable('shared', i)
            residuals = tf.cast(tf.equal(w, tf.zeros_like(w)), dtype=tf.float32)
            sw.assign(sw*residuals+w)

    def set_adapts(self, new_weights):
        for _l in range(len(self.shapes)):
            adp_kb = self.get_variable('from_kb', _l, self.curr_task)
            new_w = np.zeros_like(adp_kb.numpy())
            if len(adp_kb.shape) == 5:
                for _c in range(len(new_weights)):
                    new_w[:,:,:,:,_c] = new_weights[_c][_l]
            else:
                for _c in range(len(new_weights)):
                    new_w[:,:,_c] = new_weights[_c][_l]
            adp_kb.assign(new_w)
            
    def get_train_size(self):
        return self.train_size

    def get_task_id(self):
        return self.curr_task

    def l1_pruning(self, weights, hyp):
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
        return tf.multiply(weights, hard_threshold)

    def calculate_capacity(self):
        num_active_params = 0
        num_total_params = 0 
        for dims in self.shapes:
            params = 1
            for d in dims:
                params *= d
            num_total_params += params
        for tid in range(self.opt.num_tasks):
            num_total_params += self.shapes[-1][-1]*self.opt.num_classes
        top_most = self.heads[self.curr_task]
        top_most_kernel = top_most.kernel
        top_most_bias = top_most.bias
        var_list = self.trainable_variables.copy()
        var_list += [top_most_kernel, top_most_bias]
        for var in var_list:
            if 'adaptive' in var.name:
                var = self.l1_pruning(var, self.opt.lambda_l1)
            actives = tf.not_equal(var, tf.zeros_like(var))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()
        self.capacity.append(num_active_params/num_total_params)
        console_log('[client:%d] capacity: %.3f' %(self.client_id, num_active_params/num_total_params))

    def calculate_comm_costs(self, sw_pruned):
        num_total_params = 0 
        for i, sw in enumerate(self.variables['shared']):
            params = 1
            for d in sw.shape:
                params *= d
            num_total_params += params
        num_active_params = 0
        for i, pruned in enumerate(sw_pruned):
            actives = tf.not_equal(pruned, tf.zeros_like(pruned))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()
        self.communication.append(num_active_params/num_total_params)
        console_log('[client:%d] client->server cost: %.3f' %(self.client_id, num_active_params/num_total_params))

    def log_performance(self):
        write_file(self.opt.log_dir, 'client-{}.txt'.format(self.client_id), {
            'client_id' : self.client_id,
            'performance_task' : self.performance_task,
            'options' : vars(self.opt),
            'capacity': self.capacity,
            'cost_c': self.communication
        })