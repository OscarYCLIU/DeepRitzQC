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
    # optimizeru = optim.Opimizer(network_model.parameters(), lr = args.lr)
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
    # difference = torch.Tensor([float('inf')])
    loss_record = np.zeros(args.num_iter)
    max_iter_f = 25
    error_tol = 1e-5
    dx = .0001

    df = torch.zeros((mesh_size, args.dimension, args.dimension))
    df_tetra = torch.zeros((numf, args.dimension, args.dimension))
    lambda_ = np.zeros((numf, args.dimension, args.dimension))
    # partial_f = df_tetra.detach().numpy()

    vx_pt = Variable(mesh_pt, requires_grad=True)
    d_origin = torch.zeros((numf, args.dimension, args.dimension))
    origin_inverse = torch.zeros((numf, args.dimension, args.dimension))
    for j in range(numf):
        facet = triangulation[j, :]
        v_origin = mesh[facet, :]
        d_origin[j, :, :] = v_origin[0:3, :] - v_origin[3, :].repeat(3, 1)
        origin_inverse[j, :, :] = torch.inverse(d_origin[j, :, :]).transpose(0,1)
    r_ = df_tetra.detach().numpy()
    loss = Variable(torch.zeros(1))
    # The above is for initialization

    # The following is the first iteration
    u = network(vx_pt)
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
        d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
        df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))
    for j in range(numf):
        a = torch.norm(df_tetra[j, :, :], p=2) ** 2
        b = torch.det(df_tetra[j, :, :]) ** (2 / 3)
        kf = a / b
        loss += kf
    b1 = torch.norm(u[landmark_index, :].double() - torch.from_numpy(args.target).double(), p=2) ** 2
    b2 = 0
    for j in range(numf):
        b2 += torch.norm(
            df_tetra[j, :, :] - torch.FloatTensor(r_[j, :, :]) - torch.FloatTensor(lambda_[j, :, :])) ** 2

    loss_fpart = loss / numf
    loss_constraint = 5 * b1.type(torch.FloatTensor)
    loss = loss_fpart + loss_constraint + args.mu * b2 / numf
    # update f
    optimizeru.zero_grad()
    loss.backward(retain_graph=True)
    optimizeru.step()
    network.zero_grad()
    loss_last = loss
    print('iteration 1:')
    print(' f-subproblem loss = ', loss.detach())

    for i in range(args.num_iter):
        # R subproblem
        if not i==0:
            print('iteration :', i+1)
        # loss_f = np.zeros(max_iter_f)
        u_last = u
        u = network(vx_pt)
        difference = torch.max(torch.abs(u_last - u))

        if difference <= 1e-6:
            loss_record[i] = loss_last.detach().numpy()
            print('Finish!')
            break
        if difference > 1e-6:
            # calculating Df first
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
                d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
                df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))

            r_ = rsubproblem_np.rsubproblem(df_tetra.detach().numpy(), lambda_, eta_)
            lambda_ = lambda_ + r_ - df_tetra.detach().numpy()
            energy_r = 0
            # difference = torch.max(torch.max(torch.abs(u_last - u)))
            for iterand in range(numf):
                energy_r += np.linalg.norm(r_[iterand, :] - df_tetra[iterand, :].detach().numpy() + lambda_[iterand, :])**2
            # if i == 0:
            loss_r = args.mu * energy_r / numf + loss_fpart.detach().numpy() + loss_constraint.detach().numpy()
                # loss_r = args.mu * energy_r / numf + loss_fpart.detach().numpy()
        print('After R-subproblem, loss = ', loss_r)

        k = 0
        rel_error = torch.Tensor([float('inf')])
        while np.abs(rel_error) > error_tol and k < max_iter_f:
            # start_time = time.time()
            loss = Variable(torch.zeros(1))
            if not k==0:
                u = network(vx_pt)
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
                    d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
                    df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))

            for j in range(numf):
                a = torch.norm(df_tetra[j, :, :], p=2) ** 2
                b = torch.det(df_tetra[j, :, :]) ** (2 / 3)
                kf = a / b
                loss += kf
            # b1 is the constraint energy
            b1 = torch.norm(u[landmark_index, :].double() - torch.from_numpy(args.target).double(), p=2) ** 2
            b2 = 0
            # b2 is the Kf energy
            for j in range(numf):
                b2 += torch.norm(
                    df_tetra[j, :, :] - torch.FloatTensor(r_[j, :, :]) - torch.FloatTensor(lambda_[j, :, :])) ** 2

            loss_fpart = loss/numf + 5 * b1.type(torch.FloatTensor)
            loss = loss_fpart + args.mu * b2 / numf
            # loss_f[k] = loss.detach().numpy()
            rel_error = ((loss - loss_last)/loss).detach().numpy()
            # first_time = time.time()
            if np.mod(k, 5) == 0:
                print(k+1, ': loss of f-subproblem = ', loss.detach())
                # print('elapse time = ', first_time - start_time)
            if not k==0:
                loss_last = loss
            k += 1
            # loss_last = loss
            # f-subproblem updating
            optimizeru.zero_grad()
            loss.backward(retain_graph=True)
            optimizeru.step()
            network.zero_grad()
            second_time = time.time()
            # if np.mod(k, 5) == 0:
            #     print('BP time = ', second_time - first_time)
        loss_record[i] = loss_last.detach().numpy()
    print('loss = ', loss_last.detach())
    fid = open('result.txt','w')
    for i in range(numf):
        for j in range(4):
            fid.write(str(triangulation[i,j])+' ')
        fid.write('\n')
    fid.write('\n')
    for i in range(mesh_size):
        fid.write(' '.join(map(str, u[i,:].detach().numpy())))
        fid.write('\n')
    fid.close()
    fid1 = open('loss_record.txt','w')
    fid1.write('iteration:     energy:      \n')
    for i in range(args.num_iter):
        fid1.write(str(i+1)+'    ')
        fid1.write(' '.join(str( loss_record[i])))
        fid1.write('\n')
    fid1.close()

    return u, loss_record












