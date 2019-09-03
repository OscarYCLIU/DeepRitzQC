import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import network_model
import rsubproblem_np
from IPython import embed
import matplotlib
matplotlib.use('TkAgg')
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import rsubproblem

# matplotlib.use('macosx')


# This test uses finite difference (FDM) to compute partial derivatives
# landmark constraint is set to be a part of the energy
# Only one landmark is added
# Tetrahedron-wise Df_tetra is computed from pointwise Df, which is obtained using networks
# kind of slow in speed, 0-th version
# Repeat f-subproblem for several times


def initialization(args):
    # initialization
    network = network_model.model(args.m, args.block_num, args.dimension)
    network.apply(lambda w: network_model.weights_init(w, args.std))
    if args.adam:
        optimizeru = optim.Adam(network.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        optimizeru = optim.RMSprop(network.parameters(), lr=args.lr)
    return network, optimizeru

def training(network, optimizeru, args):
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    mesh, triangulation, landmark_index = args.Omega_mesh(args.dx, args.landmark)
    mesh_pt = torch.FloatTensor(np.hstack((mesh, np.zeros((mesh.shape[0], args.m - args.dimension)))))
    mesh_size = np.shape(mesh)[0]
    mesh = Variable(torch.FloatTensor(mesh), requires_grad = True)

    numf = np.shape(triangulation)[0]
    eta_ = 1e2 / (1/(mesh_size/3)**(2/3))
    difference = torch.Tensor([float('inf')])
    start_time = time.time()
    loss_record = np.zeros(args.num_iter)
    max_iter_f = 100
    error_tol = 1e-7

    df = torch.zeros((mesh_size, args.dimension, args.dimension))
    df_tetra = torch.zeros((numf, args.dimension, args.dimension))
    lambda_ = np.zeros((numf, args.dimension, args.dimension))
    partial_f = df_tetra.detach().numpy()
    loss_f = np.zeros(max_iter_f)

    rel_error = torch.Tensor([float('inf')])
    vx_pt = Variable(mesh_pt, requires_grad=True)
    d_origin = torch.zeros((numf, args.dimension, args.dimension))
    origin_inverse = torch.zeros((numf, args.dimension, args.dimension))
    for j in range(numf):
        facet = triangulation[j, :]
        v_origin = mesh[facet, :]
        d_origin[j, :, :] = v_origin[0:3, :] - v_origin[3, :].repeat(3, 1)
        origin_inverse[j, :, :] = torch.inverse(d_origin[j, :, :]).transpose(0,1)

    # file1 = open('result.txt', 'w')


    for i in range(args.num_iter):
        # first f-subproblem
        k = 0
        if i == 0:
            r_ = partial_f
        print('iteration :', i)
        loss_f = np.zeros(max_iter_f)
        while np.abs(rel_error) > error_tol and k < max_iter_f:
            loss = Variable(torch.zeros(1))
            u = network(vx_pt)
            dx = .0001
            for j in range(args.dimension):
                x_increment = torch.zeros(mesh_size, args.m)
                x_increment[:, j] = torch.ones(mesh_size) * dx
                x_input1 = vx_pt + x_increment
                u_j = network(x_input1)
                df_j = (u_j - u) / dx
                df[:, :, j] = df_j
            for j in range(numf):
                facet = triangulation[j, :]
                v_target = u[facet, :]
                # v_origin = mesh[facet, :]
                d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
                # d_origin = torch.FloatTensor(v_origin[0:3, :] - v_origin[3, :].repeat(3, 1))
                df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))
                partial_f[j, :, :] = df_tetra[j, :, :].detach().numpy()
            for j in range(numf):
                # auxiliary = torch.zeros((numf, args.dimension, args.dimension))
                # auxiliary[j, :, :] = torch.ones((args.dimension, args.dimension))
                a = torch.norm(df_tetra[j, :, :], p=2) ** 2
                b = torch.det(df_tetra[j, :, :]) ** (2 / 3)
                kf = a / b
                loss += kf
            b1 = torch.norm(u[landmark_index, :].double() - torch.from_numpy(args.target).double(), p=2) ** 2
            b2 = 0
            for j in range(numf):
                b2 += torch.norm(
                    df_tetra[j, :, :] - torch.FloatTensor(r_[j, :, :]) - torch.FloatTensor(lambda_[j, :, :])) ** 2

            loss_fpart = loss/numf + 5 * b1.type(torch.FloatTensor)
            loss = loss/numf + 5 * b1.type(torch.FloatTensor) + args.mu * b2 / numf
            # loss = loss / numf + 5 * b1.type(torch.FloatTensor)
            # loss_f[i] = loss + args.mu * b2 / numf
            loss_f[k] = loss.detach().numpy()
            optimizeru.zero_grad()
            loss.backward(retain_graph=True)
            optimizeru.step()
            network.zero_grad()
            if i == 0:
                loss_last = 0
            else:
                loss_last = loss_f[k-1]
            rel_error = (loss_f[k] - loss_last)/loss_f[k]
            loss_last = loss_f[k]
            # print('f-subproblem iteration: ', k)
            # print('rel_error = ', rel_error)
            # print('error difference: ', np.abs(loss_f[k-1]-loss_f[k]))
            print('loss in the ', k, '-th f-subproblem iteration = ', loss_last)
            # if not i == 0:
            #     print(loss_f)
            k += 1

        print('loss f-subproblem = ', loss_last)
        # for ll in range(max_iter_f):
        #     print('loss2 = ', loss_f[ll])
        # print('loss2 = ', loss_record[i])
        # print('loss2_1 = ', args.mu * b2 / numf)

        # R - subproblem

        if difference <= 1e-6:
            loss_record[i] = loss_last
            print(difference)
            break
        if difference > 1e-6:
            u_last = u
            u = network(vx_pt)
            r_ = rsubproblem_np.rsubproblem(partial_f, lambda_, eta_)
            lambda_ = lambda_ + r_ - partial_f
            energy_r = 0
            difference = torch.max(torch.max(torch.abs(u_last - u)))
            for iterand in range(numf):
                energy_r += np.linalg.norm(r_[iterand, :] - partial_f[iterand, :] + lambda_[iterand, :])

            rel_error = (args.mu * energy_r/numf + loss_fpart.detach().numpy() - loss_last)/loss_last
        loss_last = args.mu * energy_r/numf + loss_fpart.detach().numpy()
        print('After R-subproblem, loss = ', loss_last)

    return u, loss_record












