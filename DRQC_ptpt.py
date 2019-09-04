import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import network_model
import rsubproblem_np


# This test uses back propagation as the partial derivatives
# landmark constraint is set to be a part of the energy
# by default, the landmark is set to be the last few points (one initially)

# Pointwise Df as well as pointwise R


# Problem:
# After the first iteration, the 

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
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)

    mesh, landmark_index = args.Omega_mesh(args.dx, args.landmark)
    mesh_pt = torch.FloatTensor(np.hstack((mesh, np.zeros((mesh.shape[0], args.m - args.dimension)))))

    # numf = np.shape(triangulation)[0]
    mesh_size = np.shape(mesh)[0]
    eta_ = 1e2 / (1/(mesh_size/3)**(2/3))
    difference = torch.Tensor([float('inf')])
    df = torch.zeros((mesh_size, args.dimension, args.dimension))

    start_time = time.time()
    loss_record = np.zeros(args.num_iter)

    r_ = torch.zeros((mesh_size, args.dimension, args.dimension))
    lambda_ = r_
    max_iter_f = 150
    tol = 1e-5
    rel_error = torch.Tensor([float('inf')])
    vx_pt = Variable(mesh_pt, requires_grad=True)
    u = network(vx_pt)
    for j in range(args.dimension):
        grad_auxiliary = torch.zeros((mesh_size, 3))
        grad_auxiliary[:, j] = 1
        u.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
        df[:, j, :] = vx_pt.grad[:, :args.dimension]
        vx_pt.grad.data.zero_()

    loss = compute_loss(df, r_, lambda_, u, landmark_index, args)
    print('Initial loss = ', loss.detach())
    optimizeru.zero_grad()
    loss.backward(retain_graph=True)
    optimizeru.step()
    network.zero_grad()
    # loss_lastiteration = loss.detach()

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

        if difference <= 1e-6:
            print('Finish!')
            break
        else:
            # R-subproblem
            r_ = rsubproblem_np.rsubproblem(df.detach().numpy(), lambda_.detach().numpy(), eta_)
            r_ = torch.Tensor(r_)
            lambda_ = lambda_ + r_ - df
            loss_r = compute_loss(df, r_, lambda_, u, landmark_index, args)
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
            loss = compute_loss(df, r_, lambda_, u, landmark_index, args)
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
    loss = Variable(torch.zeros(1))
    mesh_size = np.shape(u)[0]
    for k in range(mesh_size):
        partial_f = df[k, :, :]
        kf = torch.norm(partial_f, p=2) ** 2 / (torch.det(partial_f) ** (2 / 3))
        loss += kf
    b1 = torch.norm(u[landmark_index, :].double() - torch.from_numpy(args.target).double(), p=2) ** 2
    b2 = 0
    for j in range(mesh_size):
        # a1 = df[j, :, :].detach().double()
        # a2 = r_[j, :, :].double()
        # a3 = lambda_[j, :, :].double()
        b2 += torch.norm(
            df[j, :, :].detach().double() - r_[j, :, :].double() - lambda_[j, :, :].double()) ** 2
    loss = loss.double()/mesh_size + b2 * args.mu/mesh_size + 0.5*b1
    return loss

# def f_subproblem(df, r_, lambda_, landmark_index, u, vx_pt, loss_last, max_iterf, optimizeru, network, args):
#     mesh_size = torch.Size(vx_pt)[0]
#     for j in range(3):
#         grad_auxiliary = torch.zeros((mesh_size, 3))
#         grad_auxiliary[:, j] = 1
#         u.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
#         df[:, j, :] = vx_pt.grad[:, :3]
#     loss = compute_loss(df, r_, lambda_, u, landmark_index, args)
#     loss.backward(retain_graph=True)
#     optimizeru.step()
#     network.zero_grad()
#










