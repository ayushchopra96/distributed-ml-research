##################################
# Server
# : at the beginning, it initializes
# multiple clients and global parameters
# For each communication round,
# it aggregates updates from clients
##################################
__author__ = "Wonyong Jeong, Jaehong Yoon"
__credits__ = ["Wonyong Jeong", "Jaehong Yoon"]
__email__ = "wyjeong@kaist.ac.kr, jaehong.yoon@kaist.ac.kr"
##################################

import os
import sys
import pdb
import copy
import time
import random
import threading
import atexit
import tensorflow as tf
import tensorflow.ex
from datetime import datetime

from utils.data import *
from utils.fileio import *
from .client import Client
from sklearn.metrics.pairwise import cosine_similariy

SELF_ATTENTION_CLIENT_WEIGHT = 0.3


class Server:

    def __init__(self, opt):
        self.opt = opt
        self.clients = {}
        self.threads = []
        self.communication = []
        atexit.register(self.atexit)

    def run(self):
        console_log('[server] started')
        self.start_time = time.time()
        # self.init_global_weights()
        num_clients = self.init_clients()

        self.personalized_weights = []
        for _ in range(len(num_clients)):
            for i in range(len(self.shapes)):
                self.personalized_weights.append(
                    self.initializer(self.shapes[i])
                )

        self.train_clients()

    def init_global_weights(self):
        if self.opt.base_network == 'alexnet-like':
            self.shapes = [
                (4, 4, 3, 64),
                (3, 3, 64, 128),
                (2, 2, 128, 256),
                (4096, 2048),
                (2048, 2048),
            ]
        self.global_weights = []
        self.initializer = tf.keras.initializers.VarianceScaling(
            seed=1)  # fix seed
        for i in range(len(self.shapes)):
            self.global_weights.append(
                self.initializer(self.shapes[i]).numpy())

    def init_clients(self):
        opt_copied = copy.deepcopy(self.opt)
        gpu_ids = np.arange(len(self.opt.gpu.split(','))).tolist()
        gpu_ids_real = [int(gid) for gid in self.opt.gpu.split(',')]
        gpu_clients = [int(gc) for gc in self.opt.gpu_clients.split(',')]
        self.num_clients = np.sum(gpu_clients)
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            cid_offset = 0
            for i, gpu_id in enumerate(gpu_ids):
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    console_log('[server] creating {} clients on gpu:{} ... '.format(
                        gpu_clients[gpu_id], gpu_ids_real[gpu_id]))
                    self.clients[gpu_id] = np.array([Client(
                        cid_offset+cid, opt_copied, self.get_weights()) for cid in range(gpu_clients[gpu_id])])
                    cid_offset += len(self.clients[gpu_id])
        else:
            console_log(
                '[server] creating {} clients on cpu ... '.format(gpu_clients[0]))
            self.clients[0] = np.array(
                [Client(cid, opt_copied) for cid in range(self.num_clients)])
        return sum(gpu_clients)

    def train_clients(self):
        cids = np.arange(self.num_clients).tolist()
        num_selection = int(round(self.num_clients*self.opt.frac_clients))
        for curr_round in range(self.opt.num_rounds*self.opt.num_tasks):
            selected = random.sample(cids, num_selection)  # pick clients
            console_log('[server] round:{} train clients (selected: {})'.format(
                curr_round, selected))
            self.threads = []
            selected_clients = []
            for gpu_id, gpu_clients in self.clients.items():
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    for client in gpu_clients:
                        if not client.done:
                            if client.client_id in selected:
                                selected_clients.append(client)
                                thrd = threading.Thread(
                                    target=client.train_one_round, args=(curr_round, ))
                                self.threads.append(thrd)
                                thrd.start()
            # wait all threads each round
            for thrd in self.threads:
                thrd.join()
            # update
            self.update(selected_clients)
        console_log(
            '[server] all clients have been finshed learning their tasks.')
        console_log('[server] done. ({}s)'.format(time.time()-self.start_time))
        sys.exit()

    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights

    def update(self, selected_clients):
        # client_weights = [sc.get_weights() for sc in selected_clients]
        client_weights = [self.personalized_weights[c.client_id]
                          for c in selected_clients]
        client_masks = [w[1] for w in client_weights]
        client_weights = [w[0] for w in client_weights]
        client_sizes = [sc.get_train_size() for sc in selected_clients]
        # prev_kb = self.get_weights()
        # self.fedavg(client_weights, client_sizes, client_masks)
        self.fed_per_avg(client_weights, client_sizes, client_masks)
        # _newkb = self.compute_newkb(
        #     [np.random.random(pkb.shape) for pkb in prev_kb], prev_kb, 1e-1, 1e-5, 100)
        # self.set_weights(_newkb)
        # self.calculate_comm_costs(self.get_weights())

    def compute_newkb(self, newkb, prev_kb, gdlr, l1hyp, _iter):
        for _l in range(len(prev_kb)):
            if len(prev_kb[_l].shape) != 1:
                n_active, n_full = 0, 0
                gdloss = [0., 0.]
                for iter in range(_iter):
                    gdloss[0] = 0.5 * \
                        np.sum(np.square(newkb[_l]-self.global_weights[_l]))
                    gdloss[1] = l1hyp * np.sum(np.abs(newkb[_l]-prev_kb[_l]))
                    newkb[_l] = newkb[_l] - gdlr * (
                        newkb[_l] - self.global_weights[_l] + l1hyp * np.sign(newkb[_l] - prev_kb[_l]))
                difference = newkb[_l]-prev_kb[_l]
                diff_sort = np.sort(np.abs(difference), axis=None)
                thr_difference = diff_sort[-int(
                    (1-self.opt.server_sparsity) * len(diff_sort))]
                selected = np.where(
                    np.abs(difference) >= thr_difference, newkb[_l], np.zeros(difference.shape))
                self.global_weights[_l] = selected
                n_active += np.sum(np.not_equal(selected,
                                                np.zeros(difference.shape)))
                n_full += np.prod(difference.shape)
        return newkb

    def fedavg(self, client_weights, client_sizes, client_masks=[]):
        new_weights = [np.zeros_like(w) for w in self.get_weights()]
        if self.opt.sparse_comm:
            epsi = 1e-15
            client_masks = tf.ragged.constant(client_masks, dtype=tf.float32)
            client_sizes = [tf.math.multiply(
                m, client_sizes[i]) for i, m in enumerate(client_masks)]
            total_sizes = epsi
            for _cs in client_sizes:
                total_sizes += _cs
            for c_idx, c_weights in enumerate(client_weights):  # by client
                for lidx, l_weights in enumerate(c_weights):  # by layer
                    ratio = client_sizes[c_idx][lidx]/total_sizes[lidx]
                    new_weights[lidx] += tf.math.multiply(
                        l_weights, ratio).numpy()
        else:
            total_size = np.sum(client_sizes)
            for c in range(len(client_weights)):  # by client
                _client_weights = client_weights[c]
                for i in range(len(new_weights)):  # by layer
                    new_weights[i] += _client_weights[i] * \
                        float(client_sizes[c]/total_size)
        self.set_weights(new_weights)

    def fed_per_avg(self, client_weights, client_sizes, client_masks=[]):
        sim_matrix = self.client_similarity(client_weights)
        new_personalized_weights = []
        for per_c in range(len(self.personalized_weights)):
            new_weights = [np.zeros_like(w) for w in self.get_weights()]
            for c in range(len(client_weights)):  # by client
                _client_weights = client_weights[c]
                for i in range(len(new_weights)):  # by layer
                    new_weights[i] += sim_matrix[per_c, c] * _client_weights[i]

            new_personalized_weights.append(new_weights)
        self.personalized_weights = new_personalized_weights

    def client_similarity(self, client_weights):
        sim_matrix = np.full((len(client_weights), len(
            client_weights)), SELF_ATTENTION_CLIENT_WEIGHT)
        for j in range(len(client_weights)):
            for i in range(j):
                sim_matrix[i, j] = self.cos_sim(
                    client_weights[i], client_weights[j])
                sim_matrix[j, i] = sim_matrix[i, j]

        exp_sim = np.exp(sim_matrix)
        sim_matrix = (1 - SELF_ATTENTION_CLIENT_WEIGHT) * \
            exp_sim / exp_sim.sum(dim=0)
        return sim_matrix

    def cos_sim(self, client1, client2):
        weights = np.zeros(len(client1))
        sim = np.zeros(len(client1))
        total_params = 0

        for i, (l1, l2) in enumerate(zip(client1, client2)):
            weights[i] = l1.size
            sim[i] = cosine_similariy(l1.reshape(-1), l2.reshape(-1))
            total_params += l1.size
        return weights * sim / total_params

    def calculate_comm_costs(self, new_weights):
        current_weights = self.get_weights()
        num_base_params = 0
        for i, weights in enumerate(current_weights):
            params = 1
            for d in np.shape(weights):
                params *= d
            num_base_params += params
        num_active_params = 0
        for i, nw in enumerate(new_weights):
            actives = tf.not_equal(nw, tf.zeros_like(nw))
            actives = tf.reduce_sum(tf.cast(actives, tf.float32))
            num_active_params += actives.numpy()

        self.communication.append(num_active_params/num_base_params)
        console_log('[server] server->client costs: %.3f' %
                    (num_active_params/num_base_params))

    def stop(self):
        console_log('[server] finished learning ground truth')

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        console_log('[server] all client threads have been destroyed.')
