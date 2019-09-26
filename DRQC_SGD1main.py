# import os
# import time
import numpy as np
import torch
from IPython import embed
import matplotlib
matplotlib.use('TkAgg')
torch.autograd.set_detect_anomaly(True)
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import DRQC_SGD1 as DRQC
# import DRQC_SGD_hard as DRQC
import tkMessageBox



class option:
    def __init__(self):
        self.manualSeed = 20190925
        self.lr =  1e-4 * 5
        self.randomsample = True
        #
        self.batchsize_int = 25
        self.batchsize_bdy = 20
        self.num_iter = 8
        self.report_iter = 100

        self.m = 20
        self.block_num = 4
        self.std = 1
        self.adam = 1
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.mu = 0.5

        self.dx = 0.2
        self.beta_penalty = 500
        self.gamma_penalty = 1000
        self.dimension = 3
        # self.OmegaVolume = 8
        # self.dOmegaArea = 2 * 2 * 6
        self.landmark = np.array([[.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        self.target = np.array([[0.2, 0.0, 0.0], [0.8, .0, 0.0]])


    def Sample_from_Omega(self, num_sample, index, landmark_index):
        # Taking samples and return the indices of the chosen vertices
        if num_sample > index.size:
            tkMessageBox.showerror('Too many samples', 'num_sample cannot exceed the # of interior points')
        l = index.size
        num_landmark = landmark_index.shape[0]
        count = 0
        landmark_batch_ind = np.zeros(num_landmark)
        if self.randomsample:
            tmp = np.arange(l-1)
            np.random.shuffle(tmp)
            tmp = tmp[:num_sample]
            # tmp = index[np.random.randint(0, l-1, num_sample)]
            for i in range(num_landmark):
                if num_landmark == 1:
                    if not np.isin(landmark_index, tmp):
                        tmp[num_sample - 1] = landmark_index
                        landmark_batch_ind = num_sample - 1
                    else:
                        landmark_batch_ind = np.where(tmp == landmark_index)[0]
                else:
                    if not np.isin(landmark_index[i], tmp):
                        tmp[num_sample - count - 1] = landmark_index[i]
                        count += 1
                        landmark_batch_ind[i] = num_sample - count
                    else:
                        landmark_batch_ind[i] = np.where(tmp == landmark_index[i])[0]
                # index1 = np.hstack((tmp, landmark_index))
            return tmp, landmark_batch_ind

        else:
            tkMessageBox.showerror('Not allowed', 'Restricted to mesh structure only')

    def Sample_from_dOmega(self, num_sample, index):
        if num_sample > index.shape[0]:
            tkMessageBox.showerror('Too many samples', 'num_sample cannot exceed the # of boundary points')
        l = index.size
        if self.randomsample:
            return index[np.random.randint(0, l-1, num_sample)]
        else:
            tkMessageBox.showerror('Not allowed', 'Restricted to mesh structure only')


    def Omega_mesh(self, dx, landmark):
        x, y, z = np.meshgrid(np.arange(-1,1+dx,dx), np.arange(-1,1+dx,dx), np.arange(-1,1+dx,dx))
        v = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
        # v is of size l-by-3
        l = v.shape[0]
        ll = landmark.ndim
        if not ll == 1:
            num_landmark = landmark.shape[0]
        else:
            num_landmark = 1
        landmark_index = np.zeros(num_landmark)

        for i in range(num_landmark):
            if not ll == 1:
                check = landmark[i, :] == v
            else:
                check = landmark == v
            result = np.matmul(check, np.ones((3, 1)))
            if np.isin(3, result):
                landmark_index[i] = np.argmax(result)
            else:
                v = np.vstack((v, landmark))
                landmark_index[i] = l
            l += 1
        tri = Delaunay(v)
        faces = tri.simplices
        numf = np.shape(faces)[0]
        planar = np.zeros(numf)
        for i in range(numf):
            facet = faces[i, :]
            diff = v[facet[0:3], :] - np.tile(v[facet[3], :], (3, 1))
            if not np.linalg.matrix_rank(diff) == args.dimension:
                planar[i] = 1
            # else:
            #     print(np.linalg.det(diff))
        new_face = faces[planar==0, :]
        tri.simplices = new_face
        file2 = open('mesh.txt','w')
        file2.write('triangulation: \n')
        for i in range(new_face.shape[0]):
            for j in range(4):
                file2.write(str(new_face[i, j]))
                file2.write(' ')
            file2.write('\n')
        file2.write('vertices: \n')
        for i in range(v.shape[0]):
            for j in range(3):
                file2.write(str(v[i, j]))
                file2.write(' ')
            file2.write('\n')
        file2.close()
        return v, new_face, landmark_index


args = option()
network, optimizeru = DRQC.initialization(args)
network, loss = DRQC.training(network, optimizeru, args)
