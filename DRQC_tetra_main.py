import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from scipy.spatial import Delaunay
import DRQC_tetra as DRQC


# This function mainly compares Ka Chun's code with the desired program
# dx cannot be too large as we want to acquire all the vertices
# landmark is added to the mesh artificially


class option:
    def __init__(self):
        self.manualSeed = 20190820
        self.lr = 1e-3
        self.randomsample = False
        self.num_iter = 15
        self.report_iter = 500
        self.m = 10            # number of neurons
        self.block_num = 4
        self.std = 1
        self.adam =0
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.dx = .9
        self.beta_penalty = 500
        self.dimension = 3
        self.mu = .7
        # self.landmark = torch.FloatTensor([.0, .0, .0])
        # self.target = torch.FloatTensor([.3,.3,.3])
        self.landmark = np.array([.5, .5, .5])
        self.target = np.array([.3, .3, .3])

    def Omega_mesh(self, dx, landmark):
        x, y, z = np.meshgrid(np.arange(-1,1+dx,dx), np.arange(-1,1+dx,dx), np.arange(-1,1+dx,dx))
        mesh_points = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

        l = np.shape(x)[0]
        dimension = 3
        xs = np.reshape(x, l ** dimension, 1)
        ys = np.reshape(y, l ** dimension, 1)
        zs = np.reshape(z, l ** dimension, 1)
        v = np.transpose(np.vstack((xs, ys, zs)))           # v is of size l-by-3

        check = landmark==v
        result = np.matmul(check, np.ones((dimension, 1)))
        if np.isin(dimension, result):
            landmark_index = np.argmax(result)
        else:
            v = np.vstack((v, landmark))
            landmark_index = l+1

        tri = Delaunay(v)
        faces = tri.simplices
        numf = np.shape(faces)[0]
        planar = np.zeros(numf)
        count = 0
        for i in range(numf):
            facet = faces[i, :]
            diff = v[facet[0:3], :] - np.tile(v[facet[3], :], (3, 1))
            if not np.linalg.matrix_rank(diff) == args.dimension:
                planar[i] = 1
            # else:
            #     print(np.linalg.det(diff))
        new_face = faces[planar==0, :]
        return v, new_face, landmark_index


args = option()
network, optimizeru = DRQC.initialization(args)
network, loss_record = DRQC.training(network, optimizeru, args)
