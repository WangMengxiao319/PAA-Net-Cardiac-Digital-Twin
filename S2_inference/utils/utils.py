import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

EPS = 1e-4

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)

        return pd_mu, pd_logvar

class alphaProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts (mmJSD implementation).
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weights=None):
        if weights is None:
            num_components = mu.shape[0]
            weights = (1/num_components) * torch.ones(mu.shape).to(mu.device)
    
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        weights = torch.broadcast_to(weights, mu.shape)
        pd_var = 1. / torch.sum(weights * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weights * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        
        return pd_mu, pd_logvar
    
class weightedProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weight):

        var = torch.exp(logvar) + EPS     
        weight = weight[:, None, :].repeat(1, mu.shape[1],1)
        T = 1.0 / (var + EPS)
        pd_var = 1. / torch.sum(weight * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weight * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    
    Args:
    mu (torch.Tensor): Mean of distributions. M x D for M views.
    logvar (torch.Tensor): Log of Variance of distributions. M x D for M views.
    """

    def forward(self, mu, logvar):
        mean_mu = torch.mean(mu, axis=0)
        mean_logvar = torch.mean(logvar, axis=0)
        
        return mean_mu, mean_logvar



def visualize_PC(nodes_xyz_pre, filename='PC_label.pdf'):
    # Define custom colors for labels
    # color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}
    color_dict = {
        0: '#B0B0B0',  # Gray
        1: '#8EC6E8',  # Light blue
        2: '#1F4E79'   # Dark blue
    }
    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_gd =  color_dict[0]
    

    fig, ax1 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=2)  
    ax1.set_title('Ground truth')
    ax1.set_axis_off() # Hide coordinate space 

    # Define the interaction event handler.
    def on_rotate(event):
        # Get the current rotation angles.
        elev = ax1.elev
        azim = ax1.azim
        
        # Set the view angles for both subplots.
        ax1.view_init(elev=elev, azim=azim)
        
        # Update the figure.
        fig.canvas.draw()

    # Bind the interaction event.
    fig.canvas.mpl_connect('motion_notify_event', on_rotate)

    return plt


def visualize_PC_with_twolabel_rotated(nodes_xyz_pre, labels_pre, labels_gd, filename='PC_label.pdf'):
    # Define custom colors for labels
    # color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}
    color_dict = {
        0: '#B0B0B0',  # Gray
        1: '#8EC6E8',  # Light blue
        2: '#1F4E79'   # Dark blue
    }
    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_gd = [color_dict[label] for label in labels_gd]
    colors_pre = [color_dict[label] for label in labels_pre]
    

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=2)  
    ax1.set_title('Ground truth')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=2) 
    ax2.set_title('Prediction')
    ax1.set_axis_off() # Hide coordinate space 
    ax2.set_axis_off() # Hide coordinate space

    # Define the interaction event handler.
    def on_rotate(event):
        # Get the current rotation angles.
        elev = ax1.elev
        azim = ax1.azim
        
        # Set the view angles for both subplots.
        ax1.view_init(elev=elev, azim=azim)
        ax2.view_init(elev=elev, azim=azim)
        
        # Update the figure.
        fig.canvas.draw()

    # Bind the interaction event.
    fig.canvas.mpl_connect('motion_notify_event', on_rotate)

    return plt

def visualize_PC_with_twolabel(nodes_xyz_pre, labels_pre, labels_gd, filename=None):
    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}

    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_pre = [color_dict[label] for label in labels_pre]
    colors_gd = [color_dict[label] for label in labels_gd]

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(wspace=0)
    if filename is not None:
        plt.savefig(filename)
    # plt.show()
    # plt.close(fig)
    return plt

def visualize_two_PC(nodes_xyz_pre, nodes_xyz_gd, labels, filename='PC_recon.pdf'):
    color_dict = {0: '#BCB6AE', 1: '#BCB6AE', 2: '#BCB6AE'}
    colors = [color_dict[label] for label in labels]

    df_pre = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    df_gd = pd.DataFrame(nodes_xyz_gd, columns=['x', 'y', 'z'])

    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter(df_pre['x'], df_pre['y'], df_pre['z'], c=colors, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(211, projection='3d')
    ax2.scatter(df_gd['x'], df_gd['y'], df_gd['z'], c=colors, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(hspace=0)
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

def visualize_PC_with_label(nodes_xyz, labels=1, filename='PC_label.pdf'):
    # plot in 3d using plotly
    df = pd.DataFrame(nodes_xyz, columns=['x', 'y', 'z'])
    # define custom colors for each category
    # colors = {'0': '#BCB6AE', '1': '#288596', '3': '#7D9083'}
    # colors = {'0': 'grey', '1': 'blue', '3': 'red'}
    # df['color'] = label.astype(int)
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color = 'color', color_discrete_sequence=[colors[k] for k in sorted(colors.keys())])
    # # fig = px.scatter_3d(df, x='x', y='y', z='z', color = clr_nodes, color_continuous_scale=px.colors.sequential.Viridis)
    # fig.update_traces(marker_size = 1.5)  # increase marker_size for bigger node size
    # fig.show()   
    # plotly.offline.plot(fig)
    # fig.write_image(filename) 

    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}
    # color_dict = {0: '#BCB6AE', 1: '#288596'}
    colors = [color_dict[label] for label in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=colors, s=1.5)  
    ax.set_axis_off() # Hide coordinate space
    plt.savefig(filename)
    plt.close(fig)

def save_coord_for_visualization(data, savename):
    with open('./log/' + savename+'_LVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 0]) + ',' + str(data[i, 1]) + ',' + str(data[i, 2]) + '\n')
    with open('./log/' + savename+'_epi.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 3]) + ',' + str(data[i, 4]) + ',' + str(data[i, 5]) + '\n')
    with open('./log/' + savename+'_RVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 6]) + ',' + str(data[i, 7]) + ',' + str(data[i, 8]) + '\n')

def lossplot_detailed(lossfile_train, lossfile_val, lossfile_mesh_train, lossfile_mesh_val, lossfile_KL_train, lossfile_KL_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_PC_train, lossfile_PC_val, lossfile_ecg_train, lossfile_ecg_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val):
    ax = plt.subplot(331)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(332)
    ax.set_title('MI Dice + CE loss')
    lossplot(lossfile_mesh_train, lossfile_mesh_val)

    ax = plt.subplot(333)
    ax.set_title('MI compactness loss')
    lossplot(lossfile_compactness_train, lossfile_compactness_val)

    ax = plt.subplot(334)
    ax.set_title('KL loss')
    lossplot(lossfile_KL_train, lossfile_KL_val)

    ax = plt.subplot(335)
    ax.set_title('PC recon loss')
    lossplot(lossfile_PC_train, lossfile_PC_val)

    ax = plt.subplot(336)
    ax.set_title('ECG recon loss')
    lossplot(lossfile_ecg_train, lossfile_ecg_val)

    ax = plt.subplot(337)
    ax.set_title('MI size loss')
    lossplot(lossfile_size_train, lossfile_size_val)

    ax = plt.subplot(338)
    ax.set_title('MI RVpenalty loss')
    lossplot(lossfile_RVp_train, lossfile_RVp_val)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    return plt
    # plt.savefig("img.png")
    # plt.show()
    
def lossplot_fusion(lossfile_train, lossfile_val, lossfile_mesh_train, lossfile_mesh_val,lossfile_compactness_train, lossfile_compactness_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val):
    ax = plt.subplot(331)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(332)
    ax.set_title('MI Dice + CE loss')
    lossplot(lossfile_mesh_train, lossfile_mesh_val)

    ax = plt.subplot(333)
    ax.set_title('MI compactness loss')
    lossplot(lossfile_compactness_train, lossfile_compactness_val)

    ax = plt.subplot(337)
    ax.set_title('MI size loss')
    lossplot(lossfile_size_train, lossfile_size_val)

    ax = plt.subplot(338)
    ax.set_title('MI RVpenalty loss')
    lossplot(lossfile_RVp_train, lossfile_RVp_val)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    return plt
    # plt.savefig("img.png")
    # plt.show()

def lossplot_classify(lossfile_train, lossfile_val, lossfile_mesh_train, lossfile_mesh_val, lossfile_KL_train, lossfile_KL_val, lossfile_ecg_train, lossfile_ecg_val):
    ax = plt.subplot(221)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(222)
    ax.set_title('MI classfication loss')
    lossplot(lossfile_mesh_train, lossfile_mesh_val)

    ax = plt.subplot(223)
    ax.set_title('KL loss')
    lossplot(lossfile_KL_train, lossfile_KL_val)


    ax = plt.subplot(224)
    ax.set_title('ECG recon loss')
    lossplot(lossfile_ecg_train, lossfile_ecg_val)


    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    plt.savefig("img_classify.png")
    # plt.show()

def lossplot(lossfile1, lossfile2):
    loss = np.loadtxt(lossfile1)
    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#FF7F61') # , label='train'
    plt.legend(frameon=False)

    loss = np.loadtxt(lossfile2)
    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#2C4068') # , label='val'
    plt.legend(frameon=False)
    # plt.show()
    # plt.savefig("img.png")

def ECG_visual_two(prop_data, target_ecg, filename=None, label1 = 'pred', label2 = 'true'):   
    # prop_data[target_ecg == 0.0], target_ecg[target_ecg == 0.0] = np.nan, np.nan

    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axs = plt.subplots(1, 8, constrained_layout=True, figsize=(15, 2))
    for i in range(8):
        leadName = leadNames[i]
        axs[i].plot(prop_data[i, :], color='dodgerblue', label=label1, linewidth=2)
        axs[i].plot(target_ecg[i, :], color='grey', label=label2, linewidth=2, linestyle='--')
        axs[i].set_title('Lead ' + leadName, fontsize=12)
        # axs[i].set_axis_off() 
    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    if filename is not None:
        fig.savefig(filename)
    # plt.show()
    # plt.close(fig)
    return plt


def ECG_visual_one(prop_data,  filename=None):   
    # prop_data[target_ecg == 0.0], target_ecg[target_ecg == 0.0] = np.nan, np.nan

    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axs = plt.subplots(1, 8, constrained_layout=True, figsize=(15, 2))
    for i in range(8):
        leadName = leadNames[i]
        axs[i].plot(prop_data[i, :300], color='#e8c1be', linewidth=3)
        axs[i].set_title('Lead ' + leadName, fontsize=12)
        # You can set limits
        axs[i].set_ylim([-2, 2.3])
        axs[i].set_axis_off() 
        
    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    if filename is not None:
        fig.savefig(filename)
    # plt.show()
    # plt.close(fig)
    return plt

def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None):
    """
    From Matplotlib: Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : Axes
    data : list[float]
        The intensity values for each of the 17 segments.
    seg_bold : list[int], optional
        A list with the segments to highlight.
    cmap : colormap, default: "viridis"
        Colormap for the data.
    norm : Normalize or None, optional
        Normalizer for the data.

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)  # Remove grid

    # Fill segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    # Now, draw the segment borders.  In order for the outer bold borders not
    # to be covered by inner segments, the borders are all drawn separately
    # after the segments have all been filled.  We also disable clipping, which
    # would otherwise affect the outermost segment edges.
    # Draw edges of segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))
    
def call_bulleye(data_gt, data_pred):

    # Make a figure and Axes with dimensions as desired.
    fig = plt.figure(figsize=(6, 3), layout="constrained")
    fig.get_layout_engine().set(wspace=.1, w_pad=.2)
    # fig.canvas.manager.set_window_title('Left Ventricle Bulls Eyes (AHA)')
    # fig.suptitle(
    #     'Left Ventricle Bull’s Eye (AHA)',
    #     fontsize=16,
    #     y=1.05   # Move upward slightly to avoid clipping by constrained_layout.
    # )
    axs = fig.subplots(1, 3, subplot_kw=dict(projection='polar'))

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    # cmap = mpl.cm.viridis
    cmap = mpl.cm.Blues
    max_num = max(np.max(data_gt), np.max(data_pred))
    norm = mpl.colors.Normalize(vmin=0, vmax=max_num)
    # Create an empty ScalarMappable to set the colorbar's colormap and norm.
    # The following gives a basic continuous colorbar with ticks and labels.

    # Create the 17 segment model
    bullseye_plot(axs[0], data_gt, cmap=cmap, norm=norm)
    axs[0].set_title('GT')
    bullseye_plot(axs[1], data_pred, cmap=cmap, norm=norm)
    axs[1].set_title('Pred')
    
    error = data_pred - data_gt
    vmax = np.max(np.abs(error))  # Ensure symmetry around zero.
    if vmax == 0:
        vmax = 1e-3

    cmap_err = mpl.cm.RdBu_r
    norm_err = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    bullseye_plot(
        axs[2],
        error,
        cmap=cmap_err,
        norm=norm_err
    )
    axs[2].set_title('Error Map (Pred - GT)')
    
    # --- Draw consistent colorbars. ---
    # Force one render to determine the final subplot positions.
    fig.canvas.draw() 

    # Get the subplot coordinates.
    pos0 = axs[0].get_position()
    pos1 = axs[1].get_position()
    pos2 = axs[2].get_position()

    # Define a consistent colorbar height and y-axis position.
    cbar_y = pos0.y0 - 0.1  # Below the subplots.
    cbar_height = 0.05       # Bar thickness.

    # First colorbar: span subplots 0 through 1.
    # Width = right edge of subplot 1 - left edge of subplot 0.
    cax1_width = (pos1.x1 - pos0.x0)/2
    cax1 = fig.add_axes([pos0.x0 + cax1_width/2, cbar_y, cax1_width, cbar_height])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax1,
        orientation='horizontal', 
        label='Number in each AHA segment'
    )

    # Second colorbar: below subplot 2, with exactly the same height as the first.
    cax2 = fig.add_axes([pos2.x0, cbar_y, pos2.width, cbar_height])
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap_err, norm=norm_err),
        cax=cax2,
        orientation='horizontal',
        label='Prediction Error'
    )

    return plt
if __name__ == '__main__':
    # input_data_dir = '/path/to/dataset/gt/'
    # pc = input_data_dir + 'dense_RV_endo_output_labeled_ES_pc_6003744.ply'
    # pc_volume = calculate_pointcloudvolume(pc)
    # F_visual_CV()

    log_dir = '/path/to/output/log'
    lossfile_train = log_dir + "/training_loss.txt"
    lossfile_val = log_dir + "/val_loss.txt"
    lossfile_geometry_train = log_dir + "/training_calculate_inference_loss.txt"
    lossfile_geometry_val = log_dir + "/val_calculate_inference_loss.txt"
    lossfile_compactness_train = log_dir + "/training_compactness_loss.txt"
    lossfile_compactness_val = log_dir + "/val_compactness_loss.txt"
    lossfile_KL_train = log_dir + "/training_KL_loss.txt"
    lossfile_KL_val = log_dir + "/val_KL_loss.txt"
    lossfile_PC_train = log_dir + "/training_PC_loss.txt"
    lossfile_PC_val = log_dir + "/val_PC_loss.txt"
    lossfile_ecg_train = log_dir + "/training_ecg_loss.txt"
    lossfile_ecg_val = log_dir + "/val_ecg_loss.txt"
    lossfile_RVp_train = log_dir + "/training_RVp_loss.txt"
    lossfile_RVp_val = log_dir + "/val_RVp_loss.txt"
    lossfile_size_train = log_dir + "/training_MIsize_loss.txt"
    lossfile_size_val = log_dir + "/val_MIsize_loss.txt"

    lossplot_detailed(lossfile_train, lossfile_val, lossfile_geometry_train, lossfile_geometry_val, lossfile_KL_train, lossfile_KL_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_PC_train, lossfile_PC_val, lossfile_ecg_train, lossfile_ecg_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val)
