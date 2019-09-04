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


# This function uses BP for calculating Df
# And it requires a mesh structure beforehand
# Assume the mapping is piecewise linear, so Df_tetra can be obtained from a simple matrix operation


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
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)

    mesh, triangulation, landmark_index = args.Omega_mesh(args.dx, args.landmark)
    mesh_pt = torch.FloatTensor(np.hstack((mesh, np.zeros((mesh.shape[0], args.m - args.dimension)))))
    numf = np.shape(triangulation)[0]
    mesh_size = np.shape(mesh)[0]
    eta_ = 1e2 / (1/(mesh_size/3)**(2/3))
    difference = torch.Tensor([float('inf')])
    df = torch.zeros((mesh_size, args.dimension, args.dimension))
    df_tetra = torch.zeros((numf, args.dimension, args.dimension))
    start_time = time.time()
    loss_record = np.zeros(args.num_iter)

    r_ = torch.zeros((numf, args.dimension, args.dimension))
    lambda_ = r_
    max_iter_f = 150
    tol = 1e-5
    rel_error = torch.Tensor([float('inf')])
    vx_pt = Variable(mesh_pt, requires_grad=True)
    u = network(vx_pt)

    d_origin = torch.zeros((numf, args.dimension, args.dimension))
    origin_inverse = torch.zeros((numf, args.dimension, args.dimension))
    for j in range(numf):
        facet = triangulation[j, :]
        v_origin = mesh[facet, :]
        # a = d_origin[j, :, :]
        # b = v_origin[0:3, :]
        # d = v_origin[3, :]
        # c = np.tile(v_origin[3, :],(3, 1))
        d_origin[j, :, :] = torch.FloatTensor(v_origin[0:3, :] - np.tile(v_origin[3, :],(3, 1)))
        origin_inverse[j, :, :] = torch.inverse(d_origin[j, :, :]).transpose(0,1)


    # Compute Df pointwise
    for j in range(args.dimension):
        grad_auxiliary = torch.zeros((mesh_size, 3))
        grad_auxiliary[:, j] = 1
        u.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
        df[:, j, :] = vx_pt.grad[:, :args.dimension]
        vx_pt.grad.data.zero_()
    # Compute Df tetrahedron
    for j in range(numf):
        facet = triangulation[j, :]
        v_target = u[facet, :]
        d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
        df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))

    # Calculate loss and update parameters

    loss = compute_loss(df_tetra, r_, lambda_, u, landmark_index, args)
    print('Initial loss = ', loss.detach())
    optimizeru.zero_grad()
    loss.backward(retain_graph=True)
    optimizeru.step()
    network.zero_grad()

    for i in range(args.num_iter):
        # record u in the last iteration
        u_last = u
        for j in range(args.dimension):
            # update Df
            grad_auxiliary = torch.zeros((mesh_size, 3))
            grad_auxiliary[:, j] = 1
            u.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
            df[:, j, :] = vx_pt.grad[:, :args.dimension]
            vx_pt.grad.data.zero_()
        # Compute Df tetrahedron
        for j in range(numf):
            facet = triangulation[j, :]
            v_target = u[facet, :]
            d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
            df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))

        if difference <= 1e-6:
            print('Finish!')
            break
        else:
            # R-subproblem
            r_ = rsubproblem_np.rsubproblem(df_tetra.detach().numpy(), lambda_.detach().numpy(), eta_)
            r_ = torch.Tensor(r_)
            lambda_ = lambda_ + r_ - df_tetra
            loss_r = compute_loss(df_tetra, r_, lambda_, u, landmark_index, args)
            print(loss_r.detach())

        k = 0
        rel_error = torch.Tensor([float('inf')])
        while np.abs(rel_error) > tol and k < max_iter_f:
            if k == 0:
                loss_lastiteration = loss_r.detach()
            else:
                loss_lastiteration = loss.detach()

            # Update Df
            u = network(vx_pt)
            for j in range(args.dimension):
                grad_auxiliary = torch.zeros((mesh_size, 3))
                grad_auxiliary[:, j] = 1
                u.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
                df[:, j, :] = vx_pt.grad[:, :args.dimension]
                vx_pt.grad.data.zero_()
            for j in range(numf):
                facet = triangulation[j, :]
                v_target = u[facet, :]
                d_target = v_target[0:3, :] - v_target[3, :].repeat(3, 1)
                df_tetra[j, :, :] = torch.mm(d_target, torch.FloatTensor(origin_inverse[j, :, :]))
            loss = compute_loss(df_tetra, r_, lambda_, u, landmark_index, args)
            optimizeru.zero_grad()
            loss.backward(retain_graph=True)
            optimizeru.step()
            network.zero_grad()

            # Record the loss before updating f
            rel_error = torch.abs((loss.detach() - loss_lastiteration)/loss_lastiteration)
            k += 1
            if np.mod(k, 5) == 0:
                print(k+1, loss.detach())
            # print(k)
            # # print(df.detach())
            # print(torch.max(torch.abs(df[:])))
            # print(network.__dict__)
        loss_record[i] = loss.detach()

        u_new = network(vx_pt)
        difference = torch.max(torch.max(torch.abs(u_last - u_new)))

        print('iteration #', i, 'loss = ', loss_record[i])
        # print('u = ', u.detach() )
        # print('df = ', df.detach())
    return u, loss_record




def compute_loss(df, r_, lambda_, u, landmark_index, args):
# Requires df, r_, lambda_ are Tensors
# Remember to change them correspondingly
    loss = Variable(torch.zeros(1))
    numf = df.shape[0]
    for k in range(numf):
        partial_f = df[k, :, :]
        kf = torch.norm(partial_f, p=2) ** 2 / (torch.det(partial_f) ** (2 / 3))
        loss += kf
    b1 = torch.norm(u[landmark_index, :].double() - torch.from_numpy(args.target).double(), p=2) ** 2
    b2 = 0
    for j in range(numf):
        # a1 = df[j, :, :].detach().double()
        # a2 = r_[j, :, :].double()
        # a3 = lambda_[j, :, :].double()
        b2 += torch.norm(
            df[j, :, :].detach().double() - r_[j, :, :].double() - lambda_[j, :, :].double()) ** 2
    loss = loss.double()/numf + b2 * args.mu/numf + 0.5*b1
    return loss
