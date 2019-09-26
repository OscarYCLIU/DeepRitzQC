import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import network_modelQC
from IPython import embed
import matplotlib
matplotlib.use('TkAgg')
torch.autograd.set_detect_anomaly(True)
from matplotlib import pyplot
import tkMessageBox
import scipy.io
import h5py


# from mpl_toolkits.mplot3d import Axes3D


# This file uses the idea that f = id + Network
# Boundary condition = Dirichlet
# boundary constraint as part of the energy

def initialization(args):
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    network = network_modelQC.model(args.m, args.block_num, args.dimension)
    network.apply(lambda w:network_modelQC.weights_init(w, args.std))
    if args.adam:
        optimizeru = optim.Adam(network.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        #optimizeru = optim.SGD(network.parameters(), lr=args.lr)
        optimizeru = optim.RMSprop(network.parameters(), lr=args.lr)
    return network, optimizeru

def training(network, optimizeru, args):
    count = 0
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    result_folder = os.path.join(os.getcwd(), 'result')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    current_time = time.strftime('%Y_%b_%d_%H%M%S', time.localtime())
    foldername = os.path.join(result_folder, current_time)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    mesh_folder = os.path.join(os.getcwd(), 'mesh')
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)
    strdx = 'dx'+str(args.dx)+'.mat'
    if os.path.exists(os.path.join(mesh_folder, strdx)):
        mesh = scipy.io.loadmat(os.path.join(mesh_folder, strdx))['v']
        triangulation = scipy.io.loadmat(os.path.join(mesh_folder, strdx))['faces']-1
        triangulation = triangulation.astype('int')
        landmark_index = scipy.io.loadmat(os.path.join(mesh_folder, strdx))['landmark_ind']-1
        landmark_index = landmark_index.astype('int')
        volume = scipy.io.loadmat(os.path.join(mesh_folder, strdx))['volume'].squeeze()
        tmp_neighbors_cell = scipy.io.loadmat(os.path.join(mesh_folder, strdx))['neighbors']
        neighbors_cell = np.empty(tmp_neighbors_cell.shape[0], dtype=object)

        for i in range(neighbors_cell.shape[0]):
            neighbors_cell[i] = tmp_neighbors_cell[i][0][0].astype('int')-1

        print('Existing .mat file found, creating one ring neighbors, please wait...')
        start_time = time.time()
    else:
        print('Creating mesh now, please wait...')
        start_time = time.time()
        mesh, triangulation, landmark_index = args.Omega_mesh(args.dx, args.landmark)
        volume, difference_matrix = compute_volume(triangulation, mesh)
        neighbors_cell = check_neighbors(triangulation, mesh)

    # needs to be modified
    interior_index, boundary_index = check_interior(mesh)
    # embed()
    # i1 = np.squeeze(interior_index)
    end_time = time.time()
    print('Finish meshing, elapsed time = ', end_time-start_time)

    loss_record = np.zeros(args.num_iter)

    mesh_pt = torch.FloatTensor(np.hstack((mesh, np.zeros((mesh.shape[0], args.m - args.dimension)))))
    mesh_size = mesh.shape[0]
    num_tetra = triangulation.shape[0]
    vx_pt_whole = Variable(mesh_pt, requires_grad = True)

    x_pt = torch.FloatTensor(args.batchsize_int+args.batchsize_bdy, args.m)
    vx_pt_batch = Variable(x_pt, requires_grad = True)
    D_u_whole = torch.zeros((mesh_size, args.dimension, args.dimension))
    D_u_batch = torch.zeros((args.batchsize_int+args.batchsize_bdy, args.dimension, args.dimension))
    R_pt = torch.zeros((mesh_size, args.dimension, args.dimension))
    R_tetra = torch.zeros((num_tetra, args.dimension, args.dimension))
    lambda_tetra = R_tetra
    lambda_pt = R_pt
    tol = 1e-6
    max_iter_f = 50
    loss_rep = torch.zeros(args.num_iter * max_iter_f/5  + 1)
    rel_error = torch.Tensor([float('inf')])
    eta_ = 1e2 / (1/(num_tetra/3)**(2/3))

    ## Compute derivatives initially
    u_whole = network(vx_pt_whole) + vx_pt_whole[:, :args.dimension]
    complete_name = os.path.join(foldername, 'initial' + '.txt')
    file1 = open(complete_name, 'w')
    for i in range(mesh_size):
        for j in range(3):
            file1.write(str(u_whole[i, j].detach().numpy()))
            file1.write(' ')
        file1.write('\n')
    file1.close()

    print('landmark ', u_whole[landmark_index, :].detach().numpy())
    for j in range(args.dimension):
        grad_auxiliary = torch.zeros((mesh_size, args.dimension))
        grad_auxiliary[:, j] = 1
        u_whole.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
        D_u_whole[:, j, :] = vx_pt_whole.grad[:, :args.dimension]
        vx_pt_whole.grad.data.zero_()
    R_pt = D_u_whole.detach()
    R_tetra = point2tetra(R_pt, triangulation)

    for i in range(args.num_iter):
        k = 0
        # difference = torch.Tensor([float('inf')])
        print('Solving f-subproblem now, please wait...')
        print('Iteration ', i)
        while rel_error > tol and k < max_iter_f:
            # loss = Variable(torch.zeros(1))
            # if not k == 0:
            #     ui_batch_last = ui_batch
            sample_xi, landmark_sample_ind = args.Sample_from_Omega(args.batchsize_int, interior_index, landmark_index)
            # Assume only one landmark is added only at this moment
            sample_xb = args.Sample_from_dOmega(args.batchsize_bdy, boundary_index)
            # Taking samples from Omega, returns the indices of vertices
            sample = np.hstack((sample_xi, sample_xb))
            complement = np.zeros((sample.shape[0], args.m - args.dimension))
            vx_pt_batch.data.copy_(torch.from_numpy(np.hstack((mesh[sample, :], complement))))
            net_batch = network(vx_pt_batch)
            u_batch = net_batch + vx_pt_batch[:, :args.dimension]

            start_time = time.time()
            for j in range(args.dimension):
                grad_auxiliary = torch.zeros((args.batchsize_int+args.batchsize_bdy, args.dimension))
                grad_auxiliary[:, j] = 1
                net_batch.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
                D_u_batch[:, j, :] = vx_pt_batch.grad[:, :args.dimension]
                vx_pt_batch.grad.data.zero_()
            D_ui_batch = D_u_batch[0:sample_xi.size, :]
            if not k==0:
                loss_last = loss_f
            loss1 = compute_loss1(args, D_ui_batch, R_tetra, R_pt, lambda_pt, neighbors_cell, sample_xi, volume)
            loss2 = args.gamma_penalty * torch.norm(u_batch[landmark_sample_ind, :] - torch.FloatTensor(args.target)) **2
            loss3 = args.beta_penalty * args.dx **2 * torch.norm(net_batch[sample_xi.size: sample.size, :])**2

            loss_f = loss1 + loss2 + loss3
            if not (k==0 and i==0):
                rel_error = torch.abs((loss_f - loss_last)/loss_last)
            # loss = args.OmegaVolume * () + args.beta_penalty * args.dOmegaArea * (torch.norm(ub_batch - vxb_pt)**2) \
            # + args.gamma_penalty * torch.norm(ui_batch[landmark_sample_ind, :] - args.target) **2

            loss_f.backward(retain_graph = True)
            optimizeru.step()
            network.zero_grad()
            optimizeru.zero_grad()
            if (k==0 and i==0):
                print('loss = ', k, loss_f)
                loss_rep[count] = loss_f
                count += 1
            k += 1
            # print('Iteration: ', i)

            if np.mod(k, 5)==0:
                # print('loss = , ', k, loss_f.detach().numpy()[0])
                print('loss = ', k, loss_f)
                print('Kf loss= ,', loss1.detach().numpy()[0])
                print('landmark loss= ,', loss2.detach().numpy())
                print('boundary loss = ,', loss3.detach().numpy())
                print('landmark ', u_batch[landmark_sample_ind, :].detach().numpy())
                # u_whole = network(vx_pt_whole) + vx_pt_whole[:, :args.dimension]
                # print('problem point ', u_whole[landmark_index-1, :])
                loss_rep[count] = loss_f
                count += 1
        end_time = time.time()
        print('elapsed time = ', end_time-start_time)

        net_whole = network(vx_pt_whole)
        u_whole = net_whole + vx_pt_whole[:, :args.dimension]
        for j in range(args.dimension):
            grad_auxiliary = torch.zeros((mesh_size, args.dimension))
            grad_auxiliary[:, j] = 1
            net_whole.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
            D_u_whole[:, j, :] = vx_pt_whole.grad[:, :args.dimension]
            vx_pt_whole.grad.data.zero_()
        # print('Problem point ', D_u_whole[landmark_index-1, :])
        #
        D_u_whole_tetra = point2tetra(D_u_whole, triangulation)
        # print(D_u_whole_tetra.detach().numpy())
        network.zero_grad()
        print('Converting data and Solving R-subproblem now, please wait...')
        start_time = time.time()
        R_tetra = r_subproblem(D_u_whole_tetra.detach(), lambda_tetra, eta_)
        R_pt = tetra2point(R_tetra, neighbors_cell, mesh_size, volume)
        lambda_tetra = lambda_tetra + R_tetra - D_u_whole_tetra.detach()
        lambda_pt = tetra2point(lambda_tetra, neighbors_cell, mesh_size, volume)
        # print(R_tetra.numpy())
        print('Computing loss_r, please wait...')
        loss1 = compute_loss1(args, D_u_batch, R_tetra, R_pt, lambda_pt, neighbors_cell, sample, volume)
        loss2 = args.gamma_penalty * torch.norm(u_batch[landmark_sample_ind, :] - torch.FloatTensor(args.target)) ** 2
        loss3 = args.beta_penalty * args.dx ** 2 * torch.norm(net_batch[sample_xi.size: sample.size, :]) ** 2
        loss_r = loss1 + loss2 + loss3
        loss_last = loss_r

        # loss_r = loss1 + args.gamma_penalty * torch.norm(u_batch[landmark_sample_ind, :] - (torch.FloatTensor(args.target - args.landmark))) **2 + \
        #        + args.beta_penalty * args.dx **2 * (torch.norm(ub_batch) ** 2)
        # loss_r = loss1 + args.gamma_penalty * torch.norm(u_batch[landmark_sample_ind, :].detach() - torch.FloatTensor(args.target)) ** 2 + \
        #        + args.beta_penalty * args.dx **2 * torch.norm(net_batch[sample_xi.size: sample.size, :])**2)
        end_time = time.time()
        print('After R-subproblem, loss = ', loss_r.detach().numpy()[0])
        print('Time elapsed = ', end_time - start_time)
        # lambda_pt = tetra2point(lambda_tetra, neighbors_cell, mesh_size, volume)
        loss_record[i] = loss_last.detach().numpy()
        complete_name = os.path.join(foldername, 'result' + str(i) + '.txt')
        file1 = open(complete_name, 'w')
        file1.write('iteration ' + str(i) + ': \n')
        for i in range(mesh_size):
            for j in range(3):
                file1.write(str(u_whole[i, j].detach().numpy()))
                file1.write(' ')
            file1.write('\n')
        file1.close()

    u_whole = network(vx_pt_whole) + vx_pt_whole[:, :args.dimension]
    # file1 = open('result.txt', 'w')
    # file1.write('iteration '+str(i)+': \n')
    # for i in range(mesh_size):
    #     for j in range(3):
    #         file1.write(str(u_whole[i, j].detach().numpy()))
    #         file1.write(' ')
    #     file1.write('\n')
    # file1.close()
    # file2 = open('loss.txt', 'w')
    pyplot.figure()
    pyplot.plot(loss_record, color='b')
    pyplot.title('Training Curve per 5')
    pyplot.savefig(foldername + '/Training_curvep5.png')
    pyplot.close()

    pyplot.figure()
    pyplot.plot(loss_rep, color='b')
    pyplot.title('Training Curve')
    pyplot.savefig(foldername + '/Training_curve.png')
    pyplot.close()

    return u_whole, loss_record

    # Computing R for all the tetrahedron is sort of different


def compute_loss1(args, D_u_batch, R_tetra, R_pt, lambda_pt, neighbors, sample_interior, volume):
    loss1 = Variable(torch.zeros(1))
    B = torch.zeros((sample_interior.size, 3, 3))
    A = torch.zeros(sample_interior.size)
    for i in range(sample_interior.size):
        # a = sample_interior[i]
        # b = neighbors[a]
        neighbor_i = neighbors[sample_interior[i]].astype('int')
        # lambda_i = lambda_tetra[neighbor_i, :]
        R_i = R_tetra[neighbor_i, :]
        volume_i = volume[neighbor_i]
        for j in range(volume_i.size):
            # weighted average of one ring neighborhood
            B[i, :] = B[i, :] + volume_i[j]/(torch.dot(torch.FloatTensor(volume_i), torch.ones(volume_i.size))) * R_i[j, :]
        A[i] = 2/torch.det(B[i, :].detach())**(2/3) + args.mu
        # a = .5*A[i] * torch.norm(D_u_batch)**2
        b = lambda_pt[sample_interior[i],:]
        c = R_pt[sample_interior[i],:]
        d = D_u_batch[i, :]
        f = torch.dot((b+c).flatten(), d.flatten())
        # e = torch.sum(torch.diag(D_u_batch[i, :].clone()))
        loss1 = loss1.clone() + .5*A[i] * torch.norm(D_u_batch[i, :].detach())**2 - args.mu * f \
                - A[i] * torch.sum(torch.diag(D_u_batch[i, :].detach()))
        # print(A[i])
        # print(i, .5*A[i] * torch.norm(D_u_batch[i, :].detach())**2)
        # print(args.mu * f)
        # print(A[i] * torch.sum(torch.diag(D_u_batch[i, :].detach())))
    loss = args.dx **3 * loss1
    return loss


# Maybe can be used if no bug exists
def take_derivative(input, output, args, derivative):
    for j in range(args.dimension):
        grad_auxiliary = torch.zeros((output.shape[0], args.dimension))
        grad_auxiliary[:, j] = 1
        output.backward(gradient=grad_auxiliary, retain_graph=True, create_graph=True)
        derivative[:, j, :] = input.grad[:, :args.dimension]
        input.grad.data.zero_()
    return derivative


## Version 1: Simply take average
def point2tetra(point_info, triangulation):
    # This function transform point information to tetrahedron information, dimension = 3
    # faces = tri_obj.simplices
    faces = triangulation
    numf = faces.shape[0]
    tetra_info = torch.zeros((numf, 3, 3))
    for i in range(numf):
        facet = faces[i, :]
        # a = point_info[facet, :]
        # b = torch.ones(3)
        for j in range(4):
            tetra_info[i, :] += point_info[facet[j], :]
    tetra_info /= 4
        # tetra_info[i] = torch.mm(torch.ones(3), point_info[facet, :])/4
    return tetra_info


def check_neighbors(triangulation, mesh):
    # This function returns an object array storing the tetrahedra the points are in!
    mesh_size = mesh.shape[0]
    faces = triangulation
    numofface = faces.shape[0]
    neighbor = np.empty(mesh_size, dtype= object)
    for i in range(mesh_size):
        tetra = []
        for j in range(numofface):
            facet = faces[j, :]
            if np.isin(i, facet):
                tetra = np.hstack((tetra, j))
        neighbor[i] = tetra
    return neighbor

# def check_orientation(D_u_whole):
#     mesh_size


def compute_volume(triangulation, mesh):
    faces = triangulation
    volume = np.zeros(faces.shape[0])
    difference_matrix = np.empty((faces.shape[0], 3, 3))
    for i in range(faces.shape[0]):
        facet = faces[i, :]
        v = mesh[facet, :]
        v1 = v[1, :] - v[0, :]
        v2 = v[2, :] - v[0, :]
        v3 = v[3, :] - v[0, :]
        n = np.cross(v1, v2)
        volume[i] = np.abs(np.dot(n, v3) / 3)
        difference_matrix[i, :] = np.vstack((v1, v2, v3))
    return volume, difference_matrix


def tetra2point(tetra_info, neighbor_cell, mesh_size, volume):
    # Simply take weighted average of the one ring neighborhood
    point_info = torch.zeros((mesh_size, 3, 3))
    for i in range(mesh_size):
        neighbor_i = torch.from_numpy(neighbor_cell[i].astype('int'))
        tetra_info_i = tetra_info[neighbor_i, :]
        b_i = volume[neighbor_i]
        c_i = b_i.astype('float')
        d_i = torch.Tensor(np.array(b_i))
        # volume_i = torch.from_numpy(volume[neighbor_i].astype('float'))
        volume_i = d_i
        size_i = neighbor_i.shape[0]
        volume_sum = torch.sum(volume_i)
        for j in range(size_i):
            if size_i > 1:
                a = volume_i[j].item()
                b = volume_sum
                c = tetra_info_i[j, :]
                point_info[i, :] += a/b * c
            elif size_i == 0:
                tkMessageBox.showerror('Some mesh is degenerate')
            else:
                point_info[i, :] = volume_i/volume_sum * tetra_info_i[j, :]
    return point_info


def check_interior(mesh):
    # This function returns the index for interior and boundary points
    mesh_size = mesh.shape[0]
    maximum = np.max(np.abs(mesh), axis=1)
    interior_index = []
    boundary_index = interior_index
    for i in range(mesh_size):
        if maximum[i] == 1:
            boundary_index = np.hstack((boundary_index, i))
        else:
            interior_index = np.hstack((interior_index, i))
        # print(i, interior_index)
        # print(boundary_index)
    return interior_index.flatten().astype('int'), boundary_index.flatten().astype('int')


def r_subproblem(Du_whole_tetra, lambda_, eta_):
    ## This is for the R-subproblem
    ## All are torch.Tensors
    r_ = torch.zeros(Du_whole_tetra.shape[0], 3, 3)
    max_iter_ = 100
    for i in range(Du_whole_tetra.shape[0]):
        Du_i = Du_whole_tetra[i, :]
        lambda_i = lambda_[i, :]
        a_ = 2 * torch.sum(torch.diag(Du_i)**2) / (3 * eta_)
        d_ = torch.det(Du_i) ** (2/3)
        y = torch.zeros(3)
        index = 4
        b = Du_i - lambda_i
        u, s, v = torch.svd(b)
        x = s
        # x = torch.diag(s)
        if torch.det(b) < 0:
            index = torch.argmin(x)
        diff = torch.ones(1)
        k = 0
        while torch.abs(diff) > 1e-7 and k < max_iter_:
            d_last = d_
            if index == 4:
                y[0] = .5 * (x[0] + torch.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + torch.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + torch.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 0:
                y[0] = .5 * (x[0] - torch.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + torch.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + torch.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 1:
                y[0] = .5 * (x[0] + torch.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] - torch.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + torch.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 2:
                y[0] = .5 * (x[0] + torch.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + torch.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] - torch.sqrt(x[2] ** 2 + 4 * a_ / d_))
            r = torch.det(torch.diag(y))
            d_ = ((r/d_last)**2+4*d_last)/5

            diff = torch.abs(d_last - d_)
            k += 1
        # if np.mod(i, 5) == 0:
        #     print(k)

        s = torch.diag(y)
        r_[i, :, :] = torch.matmul(torch.matmul(u, s), torch.transpose(v, 0, 1))
    return r_








