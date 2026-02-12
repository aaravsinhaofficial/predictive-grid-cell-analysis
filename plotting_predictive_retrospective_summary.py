#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 12:35:03 2026

@author: wredman
"""

import numpy as np
import matplotlib.pyplot as plt 
from visualize import rgb

# Globals 
classifications_save = True
fig_save = True

# Loading saved grid cell properties 
trajectory_style = 'straight'
folder_name = 'steps_40_batch_200_Ng_4096_relu_lr_00001_weight_decay_00001_shape_22x22_straightness_10_trajectory_style_random_walk/analysis_outputs/predictive_retrospective/'
seed = 2
data_path = '/Users/wredman/Documents/GitHub/predictive-grid-cell-analysis/Models/Random_walk/Seed ' + str(seed) + ' weight decay 1e-04/' + folder_name
X = np.load(data_path + 'final_model.pth_' + trajectory_style + '_summary_data.npz')
shifted_ratemaps = X['shifted_ratemap']
shifted_sac = X['shifted_sac']
grid_scores = X['zero_scores']
grid_scores_shift = X['scores_60']
max_scores = X['best_scores']
optimal_shifts = X['best_shift']
thresholds = X['shuffle_thresholds']
shifts = X['lag']

n_units= np.shape(grid_scores_shift)[1]

# Classifying units 
grid_minimum = 0.1
grid_ids = np.argwhere((grid_scores > grid_minimum) & (grid_scores > thresholds))
predictive_ids = np.argwhere((grid_scores < thresholds) & (max_scores > thresholds) & (optimal_shifts > 0) & (max_scores > grid_minimum))
retrospective_ids = np.argwhere((grid_scores < thresholds) & (max_scores > thresholds) & (optimal_shifts < 0) & (max_scores > grid_minimum))

# Identifying units that are too sparse 
max_sparsity = 0.90
sparsity = np.zeros(n_units)
for ii in range(n_units):
    sparsity[ii] = np.sum(shifted_ratemaps[np.argwhere(shifts == 0), :, :, ii] == 0) / (np.shape(shifted_ratemaps)[1] * np.shape(shifted_ratemaps)[2]) 
dead_unit_ids = sparsity > max_sparsity
n_dead_units = np.sum(dead_unit_ids)

# Removing any classified units that have spuriously high grid scores and/or are too sparse
spurious_gs_thresh = 1.5
good_grid_ids = []
for ii in range(len(grid_ids)):
    sorted_grid_scores_shift = np.sort(grid_scores_shift[:, grid_ids[ii]], axis = 0)
    if (np.abs(sorted_grid_scores_shift[-1]) / np.abs(sorted_grid_scores_shift[-2])) > spurious_gs_thresh or dead_unit_ids[grid_ids[ii]]:
        good_grid_ids.append(False)
    else:
        good_grid_ids.append(True)
grid_ids = grid_ids[good_grid_ids]

good_predictive_ids = []
for ii in range(len(predictive_ids)):
    sorted_predictive_scores_shift = np.sort(grid_scores_shift[:, predictive_ids[ii]], axis = 0)
    if (np.abs(sorted_predictive_scores_shift[-1]) / np.abs(sorted_predictive_scores_shift[-2])) > spurious_gs_thresh or dead_unit_ids[predictive_ids[ii]]:
        good_predictive_ids.append(False)
    else:
        good_predictive_ids.append(True)
predictive_ids = predictive_ids[good_predictive_ids]

good_retrospective_ids = []
for ii in range(len(retrospective_ids)):
    sorted_retrospective_scores_shift = np.sort(grid_scores_shift[:, retrospective_ids[ii]], axis = 0)
    if (np.abs(sorted_retrospective_scores_shift[-1]) / np.abs(sorted_retrospective_scores_shift[-2])) > spurious_gs_thresh or dead_unit_ids[retrospective_ids[ii]]:
        good_retrospective_ids.append(False)
    else:
        good_retrospective_ids.append(True)
retrospective_ids = retrospective_ids[good_retrospective_ids]

# Saving classification 
if classifications_save: 
    np.save(data_path + 'grid_ids_' + trajectory_style + '.npy', grid_ids)
    np.save(data_path + 'predictive_ids_' + trajectory_style + '.npy', predictive_ids)
    np.save(data_path + 'retrospective_ids_' + trajectory_style + '.npy', retrospective_ids)
    np.save(data_path + 'dead_unit_ids_' + trajectory_style + '.npy', dead_unit_ids)
            
# Printing how many units were identified as belonging to each class
print("Grid units: " + str(len(grid_ids)) + " / " + str(n_units - n_dead_units))
print("Predictive units: " + str(len(predictive_ids)) + " / " + str(n_units - n_dead_units))
print("Retrospective units: " + str(len(retrospective_ids)) + " / " + str(n_units - n_dead_units))

# Plotting pie chart of distribution 
plt.figure(figsize = (4, 4))
plt.pie([len(grid_ids), len(predictive_ids), len(retrospective_ids), n_units - n_dead_units - len(grid_ids) - len(predictive_ids) -len(retrospective_ids)], 
        labels = ['Grid', 'Predictive', 'Retrospective', 'None'])
if fig_save: 
    plt.savefig(data_path + 'class_distribution' + trajectory_style +'.png')
    plt.savefig(data_path + 'class_distribution.' + trajectory_style +'svg', format = 'svg')

# Plotting max grid score vs grid score at 0 
plt.figure(figsize = (6, 4))
plt.plot([np.min(grid_scores), np.max(max_scores)], [np.min(grid_scores), np.max(max_scores)], 'r--' )
plt.plot(grid_scores, max_scores, 'ko')
plt.xlabel('GS($\Delta$ = 0)')
plt.ylabel('GS($\Delta_\max$)')
if fig_save: 
    plt.savefig(data_path + 'GS_0_GS_max_' + trajectory_style +'.png')
    plt.savefig(data_path + 'GS_0_GS_max_' + trajectory_style +'.svg', format = 'svg')

# Plotting max grid score vs grid score at 0 ratio for predictive and retrospective grid units
gs_rel_inc_predictive = (max_scores[predictive_ids] - grid_scores[predictive_ids]) / np.abs(grid_scores[predictive_ids])
plt.figure(figsize = (6, 4))
n_bins = 10
x_max = 1.5
plt.hist(gs_rel_inc_predictive, bins = np.linspace(0, x_max, n_bins))
plt.ylabel('# units')
plt.xlabel('Relative GS rel. inc.')
plt.title('Predictive grid unit GS rel. increase')
if fig_save: 
    plt.savefig(data_path + 'rel_GS_increase_predictive_' + trajectory_style + '.png')
    plt.savefig(data_path + 'rel_GS_increase_predictive_' + trajectory_style + '.svg', format = 'svg')
print(str(np.sum(gs_rel_inc_predictive > x_max)) + ' units with greater rel. inc. excluded for visualization')

gs_rel_inc_retrospective = (max_scores[retrospective_ids] - grid_scores[retrospective_ids]) / np.abs(grid_scores[retrospective_ids])
plt.figure(figsize = (6, 4))
plt.hist(gs_rel_inc_retrospective, bins = np.linspace(0, x_max, n_bins))
plt.ylabel('# units')
plt.xlabel('Relative GS improvement')
plt.title('Retrospective grid unit GS rel. inc.')
if fig_save: 
    plt.savefig(data_path + 'rel_GS_increase_retrospective_' + trajectory_style +'.png')
    plt.savefig(data_path + 'rel_GS_increase_retrospective_' + trajectory_style + '.svg', format = 'svg')
print(str(np.sum(gs_rel_inc_predictive > x_max)) + ' units with greater rel. inc. excluded for visualization')

# Plotting grid score distributions for regular, predictive, and retrospective grid units
plt.figure(figsize = (6, 4))
plt.hist(grid_scores[grid_ids])
plt.ylabel('# units')
plt.xlabel('GS($\Delta = 0$)')
plt.title('Grid unit GS')
if fig_save: 
    plt.savefig(data_path + 'GS_regular_' + trajectory_style + '.png')
    plt.savefig(data_path + 'GS_regular_' + trajectory_style + '.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(max_scores[predictive_ids])
plt.ylabel('# units')
plt.xlabel('GS($\Delta_\max$)')
plt.title('Predictive grid unit GS')
if fig_save: 
    plt.savefig(data_path + 'GS_predictive_' + trajectory_style + '.png')
    plt.savefig(data_path + 'GS_predictive_' + trajectory_style + '.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(max_scores[retrospective_ids])
plt.ylabel('# units')
plt.xlabel('GS($\Delta_\max$)')
plt.title('Retrospective grid unit GS')
if fig_save: 
    plt.savefig(data_path + 'GS_retrospective_' + trajectory_style + '.png')
    plt.savefig(data_path + 'GS_retrospective_' + trajectory_style + '.svg', format = 'svg')

# Plotting GS vs shift for all classified units 
plt.figure(figsize = (4, 6))
sorted_ids = np.argsort(optimal_shifts[grid_ids], axis = 0)
plt.imshow(grid_scores_shift[:, grid_ids[sorted_ids].flatten()].T, aspect='auto', cmap = 'jet')
plt.ylabel('Units')
plt.xlabel('Shifts')
plt.title('Grid cells')
plt.clim([0, 1.25])
plt.colorbar()
if fig_save: 
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_grid_units.png')
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_grid_units.svg', format = 'svg')

plt.figure(figsize = (4, 6))
sorted_ids = np.argsort(optimal_shifts[predictive_ids], axis = 0)
plt.imshow(grid_scores_shift[:, predictive_ids[sorted_ids].flatten()].T, aspect='auto', cmap = 'jet')
plt.ylabel('Units')
plt.xlabel('Shifts')
plt.title('Predictive cells')
plt.clim([0.0, 1.25])
if fig_save: 
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_predictive_units.png')
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_predictive_units.svg', format = 'svg')
    
plt.figure(figsize = (4, 6))
sorted_ids = np.argsort(optimal_shifts[retrospective_ids], axis = 0)
plt.imshow(grid_scores_shift[:, retrospective_ids[sorted_ids].flatten()].T, aspect='auto', cmap = 'jet')
plt.ylabel('Units')
plt.xlabel('Shifts')
plt.title('Retrospecitve cells')
plt.clim([0, 1.25])
if fig_save: 
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_retrospective_units.png')
    plt.savefig(data_path + 'GS_vs_shift_' + trajectory_style + '_retrospective_units.svg', format = 'svg')

# Plotting average grid score per shift for all classified units 
plt.figure(figsize = (6, 4))
plt.fill_between(shifts, np.mean(grid_scores_shift[:, grid_ids.flatten()], axis = 1) - np.std(grid_scores_shift[:, grid_ids.flatten()], axis = 1)/np.sqrt(len(grid_ids)), np.mean(grid_scores_shift[:, grid_ids.flatten()], axis = 1) + np.std(grid_scores_shift[:, grid_ids.flatten()], axis = 1)/np.sqrt(len(grid_ids)), alpha = 0.5)
plt.plot(shifts, np.mean(grid_scores_shift[:, grid_ids.flatten()], axis = 1))
plt.xlabel('Shifts')
plt.ylabel('Grid score')
plt.title('Grid cells')
if fig_save: 
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_grid_units.png')
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_grid_units.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.fill_between(shifts, np.mean(grid_scores_shift[:, predictive_ids.flatten()], axis = 1) - np.std(grid_scores_shift[:, predictive_ids.flatten()], axis = 1)/np.sqrt(len(predictive_ids)), np.mean(grid_scores_shift[:, predictive_ids.flatten()], axis = 1) + np.std(grid_scores_shift[:, predictive_ids.flatten()]/np.sqrt(len(predictive_ids)), axis = 1), alpha = 0.5)
plt.plot(shifts, np.mean(grid_scores_shift[:, predictive_ids.flatten()], axis = 1))
plt.xlabel('Shifts')
plt.ylabel('Grid score')
plt.title('Predictive cells')
if fig_save: 
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_predictive_units.png')
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_predictive_units.svg', format = 'svg')


plt.figure(figsize = (6, 4))
plt.fill_between(shifts, np.mean(grid_scores_shift[:, retrospective_ids.flatten()], axis = 1) - np.std(grid_scores_shift[:, retrospective_ids.flatten()], axis = 1)/np.sqrt(len(retrospective_ids)), np.mean(grid_scores_shift[:, retrospective_ids.flatten()], axis = 1) + np.std(grid_scores_shift[:, retrospective_ids.flatten()], axis = 1)/np.sqrt(len(retrospective_ids)), alpha = 0.5)
plt.plot(shifts, np.mean(grid_scores_shift[:, retrospective_ids.flatten()], axis = 1))
plt.xlabel('Shifts')
plt.ylabel('Grid score')
plt.title('Retrospective cells')
if fig_save: 
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_retrospective_units.png')
    plt.savefig(data_path + 'mean_GS_vs_shift_' + trajectory_style + '_retrospective_units.svg', format = 'svg')

# Plotting optimal shift for all classified units 
plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[grid_ids])
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Grid cells')
if fig_save: 
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_grid_units.png')
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_grid_units.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[predictive_ids])
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Predictive cells')
if fig_save: 
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_predictive_units.png')
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_predictive_units.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.hist(optimal_shifts[retrospective_ids])
plt.xlabel('Optimal shift')
plt.ylabel('Count')
plt.title('Retrospective cells')
if fig_save: 
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_retrospective_units.png')
    plt.savefig(data_path + 'optimal_shift_' + trajectory_style + '_retrospective_units.svg', format = 'svg')

# Plotting example regular grid cell 
grid_cell_plot_percentile = 99
grid_cell_plot_id = grid_ids[np.argmin(np.abs(grid_scores[grid_ids] - np.percentile(grid_scores[grid_ids], grid_cell_plot_percentile)))]

plt.figure(figsize = (6, 4))
plt.plot(shifts, grid_scores_shift[:, grid_cell_plot_id], 'k-')
plt.xlabel('$\Delta$')
plt.ylabel('GS')
if fig_save: 
    plt.savefig(data_path + 'example_grid_unit_score_vs_shift_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_grid_unit_score_vs_shift_' + trajectory_style + '.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.subplot(2, 3, 1)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == -5), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = -5')
plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 0), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = 0')
plt.subplot(2, 3, 3)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 5), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = +5')
plt.subplot(2, 3, 4)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == -5), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = -5')
plt.subplot(2, 3, 5)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 0), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = 0')
plt.subplot(2, 3, 6)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 5), :, :, grid_cell_plot_id])))
plt.title('$\Delta$ = +5')

if fig_save: 
    plt.savefig(data_path + 'example_grid_unit_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_grid_unit_' + trajectory_style + '.svg', format = 'svg')


# Plotting example predictive grid cell 
predictive_cell_plot_percentile = 99
predictive_cell_plot_id = 1905 # predictive_ids[np.argmin(np.abs(gs_rel_inc_predictive - np.percentile(gs_rel_inc_predictive, predictive_cell_plot_percentile)))]

plt.figure(figsize = (6, 4))
plt.plot(shifts, grid_scores_shift[:, predictive_cell_plot_id], 'k-')
plt.xlabel('$\Delta$')
plt.ylabel('GS')
if fig_save: 
    plt.savefig(data_path + 'example_predictive_grid_unit_score_vs_shift_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_predictive_grid_unit_score_vs_shift_' + trajectory_style + '.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.subplot(2, 3, 1)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 0), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = 0')
plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 7), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = +7')
plt.subplot(2, 3, 3)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == optimal_shifts[predictive_cell_plot_id]), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = +' + str(optimal_shifts[predictive_cell_plot_id]))
plt.subplot(2, 3, 4)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 0), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = 0')
plt.subplot(2, 3, 5)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 7), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = +7')
plt.subplot(2, 3, 6)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == optimal_shifts[predictive_cell_plot_id]), :, :, predictive_cell_plot_id])))
plt.title('$\Delta$ = +' + str(optimal_shifts[predictive_cell_plot_id]))

if fig_save: 
    plt.savefig(data_path + 'example_predictive_grid_unit_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_predictive_grid_unit_' + trajectory_style + '.svg', format = 'svg')

# Plotting example retrospective grid cell 
retrospective_cell_plot_percentile = 99
retrospective_cell_plot_id = 657 # retrospective_ids[np.argmin(np.abs(gs_rel_inc_retrospective - np.percentile(gs_rel_inc_retrospective, retrospective_cell_plot_percentile)))]

plt.figure(figsize = (6, 4))
plt.plot(shifts, grid_scores_shift[:, retrospective_cell_plot_id], 'k-')
plt.xlabel('$\Delta$')
plt.ylabel('GS')
if fig_save: 
    plt.savefig(data_path + 'example_retrospective_grid_unit_score_vs_shift_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_retrospective_grid_unit_score_vs_shift_' + trajectory_style + '.svg', format = 'svg')

plt.figure(figsize = (6, 4))
plt.subplot(2, 3, 1)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == optimal_shifts[retrospective_cell_plot_id]), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = ' + str(optimal_shifts[retrospective_cell_plot_id]))
plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == -7), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = -7')
plt.subplot(2, 3, 3)
plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 0), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = 0')
plt.subplot(2, 3, 4)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == optimal_shifts[retrospective_cell_plot_id]), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = ' + str(optimal_shifts[retrospective_cell_plot_id]))
plt.subplot(2, 3, 5)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == -7), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = -7')
plt.subplot(2, 3, 6)
plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 0), :, :, retrospective_cell_plot_id])))
plt.title('$\Delta$ = 0')

if fig_save: 
    plt.savefig(data_path + 'example_retrospective_grid_unit_' + trajectory_style + '.png')
    plt.savefig(data_path + 'example_retrospective_grid_unit_' + trajectory_style +'.svg', format = 'svg')


# Plotting all predictive cells 
plot_all_pred = False

if plot_all_pred: 
    for ii in range(len(predictive_ids)):
        plt.figure(figsize = (8, 6))
        plt.title('Unit ' + str(predictive_ids[ii]))
        plt.axis('off')
        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 0), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = 0')
        plt.subplot(2, 3, 2)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 7), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = + 7')
        plt.subplot(2, 3, 3)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == optimal_shifts[predictive_ids[ii]]), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = +' + str(optimal_shifts[predictive_ids[ii]]))
        plt.subplot(2, 3, 4)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 0), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = 0')
        plt.subplot(2, 3, 5)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 7), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = +7')
        plt.subplot(2, 3, 6)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == optimal_shifts[predictive_ids[ii]]), :, :, predictive_ids[ii]])))
        plt.title('$\Delta$ = +' + str(optimal_shifts[predictive_ids[ii]]))
    
        plt.show()
    
        input("Press Enter to continue...") # Pause for user input in the console
        print("User pressed Enter, program continues.")
        plt.close() # Close the figure


# Plotting all retrospective cells 
plot_all_retro = False

if plot_all_retro: 
    for ii in range(len(retrospective_ids)):
        plt.figure(figsize = (8, 6))
        plt.title('Unit ' + str(retrospective_ids[ii]))
        plt.axis('off')
        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == 0), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = 0')
        plt.subplot(2, 3, 2)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == -7), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = -7')
        plt.subplot(2, 3, 3)
        plt.imshow(np.squeeze(rgb(shifted_ratemaps[np.argwhere(shifts == optimal_shifts[retrospective_ids[ii]]), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = ' + str(optimal_shifts[retrospective_ids[ii]]))
        plt.subplot(2, 3, 4)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == 0), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = 0')
        plt.subplot(2, 3, 5)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == -7), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = -7')
        plt.subplot(2, 3, 6)
        plt.imshow(np.squeeze(rgb(shifted_sac[np.argwhere(shifts == optimal_shifts[retrospective_ids[ii]]), :, :, retrospective_ids[ii]])))
        plt.title('$\Delta$ = ' + str(optimal_shifts[retrospective_ids[ii]]))
    
        plt.show()
    
        input("Press Enter to continue...") # Pause for user input in the console
        print("User pressed Enter, program continues.")
        plt.close() # Close the figure



