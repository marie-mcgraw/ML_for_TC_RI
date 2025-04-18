import numpy as np
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,classification_report

def make_CM_plot(cm,labels,ax_sel,tick_labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    #
    disp.plot(ax=ax_sel,cmap='magma_r')
    ax_sel.set_xlabel('Predicted',fontsize=16)
    ax_sel.set_ylabel('Actual',fontsize=16)
    ax_sel.set_xticklabels(tick_labels,fontsize=16)
    ax_sel.set_yticklabels(tick_labels,fontsize=16)
##
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,ax_sel):
   ## """
    ##Modified from:
    ##Hands-On Machine learning with Scikit-Learn
    ##and TensorFlow; p.89
    ##"""
    # plt.figure(figsize=(8, 8))
    ax_sel.set_title("Precision and Recall Scores as a function of the decision threshold")
    ax_sel.plot(thresholds, precisions[:-1], "b--", label="Precision",linewidth=3)
    ax_sel.plot(thresholds, recalls[:-1], "g-", label="Recall",linewidth=3)
    ax_sel.set_ylabel("Score")
    ax_sel.set_xlabel("Decision Threshold")
    ax_sel.legend(loc='best')
####
def plot_roc_curve(fpr, tpr, ax_sel, label=None):
    ##"""
    ##The ROC curve, modified from 
    ##Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    ##"""
    # plt.figure(figsize=(8,8))
    ax_sel.set_title('ROC Curve')
    ax_sel.plot(fpr, tpr, linewidth=3, label=label)
    ax_sel.plot([0, 1], [0, 1], 'k--')
    #ax_sel.axis([-0.005, 1, 0, 1.005])
    ax_sel.set_xlim([-0.005,1.005])
    ax_sel.set_ylim([-0.005,1.005])
    ax_sel.set_xlabel("False Positive Rate")
    ax_sel.set_ylabel("True Positive Rate (Recall)")
    ax_sel.legend(loc='best')
## 
def precision_recall_threshold(p, r, thresholds, y_scores, y_test, t=0.5):
    ##"""
   ## plots the precision recall curve and shows the current value for each
   ## by identifying the classifier's threshold (t).
   ## """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
### Performance diagram plotting code. Barely adapted from Ryan Lagerquist's plotting evaluation code. 
#1. Create a grid 
def _get_sr_pod_grid(success_ratio_spacing=0.01, pod_spacing=0.01):
    """Creates grid in SR-POD space
    SR = success ratio
    POD = probability of detection
    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid
    :param success_ratio_spacing: Spacing between adjacent success ratios
        (x-values) in grid.
    :param pod_spacing: Spacing between adjacent POD values (y-values) in grid.
    :return: success_ratio_matrix: M-by-N numpy array of success ratios.
        Success ratio increases while traveling right along a row.
    :return: pod_matrix: M-by-N numpy array of POD values.  POD increases while
        traveling up a column.
    """

    num_success_ratios = int(np.ceil(1. / success_ratio_spacing))
    num_pod_values = int(np.ceil(1. / pod_spacing))

    unique_success_ratios = np.linspace(
        0, 1, num=num_success_ratios + 1, dtype=float
    )
    unique_success_ratios = (
        unique_success_ratios[:-1] + success_ratio_spacing / 2
    )

    unique_pod_values = np.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return np.meshgrid(unique_success_ratios, unique_pod_values[::-1])
# 2. Get frequency bias from success ratio and POD
def _bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: frequency_bias_array: numpy array (same shape) of frequency biases.
    """

    return pod_array / success_ratio_array
# 3. Get CSI / Threat Score from success ratio and POD
def _csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: csi_array: numpy array (same shape) of CSI values.
    """

    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1
# 4.  Get the colormap for the CSI (shaded contours on performance diagram)
def _get_csi_color_scheme(cmap_use = 'GnBu'):
    """Returns colour scheme for CSI (critical success index). Default cmap is GnBu but can be changed if desired. 
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """
    CSI_LEVELS = np.linspace(0,1,11,dtype=float)
    this_colour_map_object = plt.get_cmap(cmap_use)
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, this_colour_map_object.N
    )
    #
    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        CSI_LEVELS
    ))
    #
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]
    #
    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, colour_map_object.N
    )
    return colour_map_object, colour_norm_object
# 5.  Get the colorbar object (will refer to CSI contours)
def _add_color_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value=None,
        max_colour_value=None, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True):
    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = plt.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = plt.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=0.95
    )

    colour_bar_object.ax.tick_params(labelsize=14)
    return colour_bar_object
# 6. Get the basic PD plot--x axis should be success ratio, y axis should be probability of detection (POD).  We'll add shading for the CSI / Threat Score, and diagonal lines corresponding to the bias. We'll return this figure and add our actual results separately. 
def make_performance_diagram_background(ax):
    # Defaults for diagram
    CSI_LEVELS = np.linspace(0,1,11,dtype=float)
    # PEIRCE_levels = np.linspace(0,1,11,dtype=float)
    FREQ_BIAS_PADDING = 10
    FREQ_BIAS_STRING_FORMAT = '%.2f'
    PD_color = np.array([228,26,28],dtype=float)/255
    # Get POD / SR grid; calculate CSI grid from SR/POD grid; and calculate bias grid from SR and POD grid
    success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
    csi_matrix = _csi_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix)
    frequency_bias_matrix = _bias_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix)
    # Get the colormap for CSI
    this_color_map_object, this_color_norm_object = (
            _get_csi_color_scheme())
    # Add CSI contours and corresponding colorbar
    ax.contourf(success_ratio_matrix, pod_matrix, csi_matrix,
              cmap=this_color_map_object, norm=this_color_norm_object,vmin=0.,alpha=0.5,
            vmax=1., axes=ax)
    
    colour_bar_object = _add_color_bar(
            axes_object=ax, values_to_colour=csi_matrix,
            colour_map_object=this_color_map_object,
            colour_norm_object=this_color_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False)
    colour_bar_object.set_label('CSI (critical success index)',fontsize=18)
    # Set up defaults for frequency bias lines
    # FREQ_BIAS_COLOUR = np.full(3, 152. / 255)
    FREQ_BIAS_LEVELS = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])
    # FREQ_BIAS_WIDTH = 2
    # Now, add the lines of constant bias and label them. 
    bias_contour_object = ax.contour(
        success_ratio_matrix, pod_matrix, frequency_bias_matrix,
        FREQ_BIAS_LEVELS,axes=ax,cmap=plt.get_cmap('copper_r'),linewidths=4)
    ax.clabel(
        bias_contour_object, inline=True, inline_spacing=FREQ_BIAS_PADDING,
        fmt=FREQ_BIAS_STRING_FORMAT, fontsize=17)
    # Label axes
    ax.set_ylabel(r"Probability of Detection (${\frac{hits}{hits+misses}})$",fontsize=21)
    ax.set_xlabel(r"1 - False Alarm Ratio (${\frac{False Alarms}{Hits + False Alarms}})$",fontsize=21)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=22)
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=22)
    return ax
# 7. Add model results to performance diagram
def add_model_results(ax,CM,cat_sel='RI'):
    # Inputs:
    # ax: axes for corresponding figure
    # CM: dataframe containing confusion matrix 
    # Category selected: RI or not RI? Default is RI
    # 
    # Get the means, minima, and maxima for each model / basin
    CM_rs = CM.reset_index().set_index(['Category Names','BASIN','Model','Fold'])
    CM_mean = CM_rs.median(level=(0,1,2))
    CM_min = CM.groupby(['Category Names','BASIN','Model']).quantile(0.025)
    CM_max = CM.groupby(['Category Names','BASIN','Model']).quantile(0.975)
    CM_ALL_sel = CM_mean.loc[cat_sel]
    # Get error bar bounds
    CM_mean_sort = CM_ALL_sel.reset_index().sort_values(['BASIN','Model'],ignore_index=True)
    CM_min_sort = CM_min.loc[cat_sel].reset_index().sort_values(['BASIN','Model'],ignore_index=True)
    CM_max_sort = CM_max.loc[cat_sel].reset_index().sort_values(['BASIN','Model'],ignore_index=True)
    y_lower = CM_mean_sort['POD'] - CM_min_sort['POD']
    # y_lower = CM_min_sort['POD']
    y_upper = CM_max_sort['POD'] - CM_mean_sort['POD']
    # y_upper = CM_max_sort['POD']
    x_lower = CM_mean_sort['SR'] - CM_min_sort['SR']
    x_upper = CM_max_sort['SR'] - CM_mean_sort['SR']
    # Make a colormap for each basin
    colors_list = ['hot pink','navy','goldenrod','green','violet']
    pal_sel = sns.color_palette(sns.xkcd_palette(colors_list),5)
    # Plot error bars in both x and y
    ax.errorbar(CM_ALL_sel.reset_index().sort_values(['BASIN','Model'],ignore_index=True)['SR'],
                CM_ALL_sel.reset_index().sort_values(['BASIN','Model'],ignore_index=True)['POD'],
                yerr=[y_lower,y_upper],xerr=[x_lower,x_upper],
           linestyle='none',linewidth=2,color='xkcd:slate grey',zorder=9)
    sns.scatterplot(data=CM_ALL_sel.reset_index().sort_values(['BASIN','Model'],ignore_index=True),
                  x='SR',y='POD',hue='BASIN',style='Model',ax=ax,
                   palette=sns.set_palette(pal_sel),s=300,alpha=0.8,zorder=10)
    ax.legend(fontsize=20).set_zorder(10)
# 8.  Plot basic score vs basin.  If we want to see spread in some kind of score value as a function of basin. Options for box plot or swarm plot
def plot_basic_score_basin(ax,data,y_val,is_swarm=True):
    if is_swarm:
        sns.swarmplot(data=data,x='BASIN',y=y_val,ax=ax,s=12)
    else:
        sns.boxplot(data=data,x='BASIN',y=y_val,ax=ax)
    ax.set_xlabel(None)
    ax.set_xticklabels(data.reset_index()['BASIN'].unique().tolist(),fontsize=16,rotation=30)
    #ax.set_ylabel('Area Under PD Curve',fontsize=16)
    ax.tick_params(axis='y',labelsize=14)

#9. Plot CSI vs bias
def plot_CSI_vs_bias(p_vs_r,ax):
    # Get bias and CSI for maximum values of CSI
    ax.axvline(1,color='xkcd:charcoal',linewidth=2)
    b_c_max = p_vs_r.sort_values(['CSI'], ascending=[False]).groupby(['BASIN','Fold']).first()
    # Plot
    sns.scatterplot(data=b_c_max.reset_index(),x='Bias',y='CSI',hue='BASIN',ax=ax,alpha=0.4,legend=False)
    # Get mean bias for max CSI of each basin
    sns.scatterplot(data=b_c_max.mean(level=0).reset_index(),
                x='Bias',y='CSI',hue='BASIN',s=200,ax=ax)
    # Formatting
    ax.legend(fontsize=13)
    ax.tick_params(axis='y',labelsize=14)
    ax.tick_params(axis='x',labelsize=14)
    ax.set_ylabel('CSI',fontsize=16)
    ax.set_xlabel('Bias',fontsize=16)
    ax.grid()
    ax.set_xlim([0.3,4])
    ax.text(0.4,0.57,'Under-',fontsize=13)
    ax.text(0.4,0.55,'predicting RI',fontsize=13)
    ax.text(2,0.57,'Overpredicting RI',fontsize=13)
# 10. Plot SR / POD Curves on performance diagram
def plot_PD_curves(p_vs_r,ax,basin_sel,style='-',plot_folds=True):
    pr_smoothed = p_vs_r.groupby(['BASIN','Fold','Thresh Round']).mean().xs(basin_sel)
    pr_mean = pr_smoothed.groupby(['Thresh Round']).mean().reset_index()

    # Curve for each fold
    if plot_folds:
        folds = p_vs_r['Fold'].unique()
        for ifold in folds:
            pr_smoothed.xs(ifold).plot(x='Success Ratio',y='POD',color='xkcd:purple',ax=ax,alpha=0.25,label='_nolegend_')
    # Get curve for median AUPD
    max_CSI_ind = p_vs_r.groupby(['BASIN','Fold'])[['CSI','Bias']].agg({'CSI':'max'})#.xs(basin_sel)
    aupd_med = max_CSI_ind.xs(basin_sel).median()
    med_fold = max_CSI_ind.xs(basin_sel).loc[max_CSI_ind.xs(basin_sel)['CSI']==aupd_med[0]]
    min_fold = max_CSI_ind.xs(basin_sel).idxmin()
    max_fold = max_CSI_ind.xs(basin_sel).idxmax()
    pr_smoothed.xs(med_fold.index[0]).plot(x='Success Ratio',y='POD',ax=ax,color='xkcd:purple',linewidth=6,linestyle=style,
                                           label='Median CSI')
    pr_smoothed.xs(min_fold[0]).plot(x='Success Ratio',y='POD',ax=ax,color='xkcd:magenta',linewidth=6,linestyle=style,
                                           label='Min CSI')
    pr_smoothed.xs(max_fold[0]).plot(x='Success Ratio',y='POD',ax=ax,color='xkcd:crimson',linewidth=6,linestyle=style,
                                           label='Max CSI')
    #sns.lineplot(data=pr_mean.sort_values('POD'),x='Success Ratio',y='POD',hue='Fold',
     #           sort=False,color='xkcd:purple',ax=ax,linewidth=4,legend=False)
    ax.set_xlim([0,1])
    ax.legend(fontsize=17)
    ax.set_ylim([0,1])
#
# 10. Plot SR / POD Curves on performance diagram
def plot_PD_curves_compare_models(p_vs_r,ax,basin_sel,CSI_metric='median'):
    pr_smoothed = p_vs_r.groupby(['BASIN','Model','Fold','Thresh Round']).mean().xs(basin_sel)
    #pr_mean = pr_smoothed.groupby(['Model','Thresh Round']).mean().reset_index()
    # Get curve for median AUPD
    max_CSI_ind = p_vs_r.groupby(['BASIN','Model','Fold'])[['CSI','Bias']].agg({'CSI':'max'})#.xs(basin_sel)
    colors = ['xkcd:coral','xkcd:turquoise','xkcd:tangerine','xkcd:leaf green']
    if CSI_metric == 'median':
        aupd_med = max_CSI_ind.xs(basin_sel).median(level=0)
        models_list = aupd_med.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
           # print(i_model)
            CSI_fold = max_CSI_ind.xs((basin_sel,i_model)).loc[max_CSI_ind.xs((basin_sel,
                                                   i_model))['CSI']==aupd_med.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=colors[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
        
    elif CSI_metric == 'min':
        min_fold = max_CSI_ind.xs(basin_sel).min(level=0)
        models_list = min_fold.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
            # print(i_model)
            CSI_fold = max_CSI_ind.xs((basin_sel,i_model)).loc[max_CSI_ind.xs((basin_sel,
                                                   i_model))['CSI']==min_fold.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=colors[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
    elif CSI_metric == 'max':
        min_fold = max_CSI_ind.xs(basin_sel).max(level=0)
        models_list = min_fold.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
           #  print(i_model)
            CSI_fold = max_CSI_ind.xs((basin_sel,i_model)).loc[max_CSI_ind.xs((basin_sel,
                                                   i_model))['CSI']==min_fold.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=colors[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
    #sns.lineplot(data=pr_mean.sort_values('POD'),x='Success Ratio',y='POD',hue='Fold',
     #           sort=False,color='xkcd:purple',ax=ax,linewidth=4,legend=False)
    ax.set_xlim([0,1])
    ax.legend(fontsize=17)
    ax.set_ylim([0,1])
# 10b. Plot SR / POD Curves on performance diagram
def plot_PD_curves_compare_basins(p_vs_r,ax,model_sel,CSI_metric='median'):
    pr_smoothed = p_vs_r.groupby(['Model','BASIN','Fold','Thresh Round']).mean().xs(model_sel)
    #pr_mean = pr_smoothed.groupby(['Model','Thresh Round']).mean().reset_index()
    # Get curve for median AUPD
    max_CSI_ind = p_vs_r.groupby(['Model','BASIN','Fold'])[['CSI','Bias']].agg({'CSI':'max'})#.xs(basin_sel)
    colors_list = ['hot pink','navy','goldenrod','green','violet']
    pal_sel = sns.color_palette(sns.xkcd_palette(colors_list),5)
    if CSI_metric == 'median':
        aupd_med = max_CSI_ind.xs(model_sel).median(level=0)
        models_list = aupd_med.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
           # print(i_model)
            CSI_fold = max_CSI_ind.xs((model_sel,i_model)).loc[max_CSI_ind.xs((model_sel,
                                                   i_model))['CSI']==aupd_med.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=pal_sel[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
        
    elif CSI_metric == 'min':
        min_fold = max_CSI_ind.xs(model_sel).min(level=0)
        models_list = min_fold.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
            # print(i_model)
            CSI_fold = max_CSI_ind.xs((model_sel,i_model)).loc[max_CSI_ind.xs((model_sel,
                                                   i_model))['CSI']==min_fold.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=pal_sel[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
    elif CSI_metric == 'max':
        min_fold = max_CSI_ind.xs(model_sel).max(level=0)
        models_list = min_fold.index.unique().tolist()
        for i in np.arange(0,len(models_list)):
            i_model = models_list[i]
           #  print(i_model)
            CSI_fold = max_CSI_ind.xs((model_sel,i_model)).loc[max_CSI_ind.xs((model_sel,
                                                   i_model))['CSI']==min_fold.loc[i_model]['CSI']]
            pr_smoothed.xs((i_model,CSI_fold.index[0])).plot(x='Success Ratio',y='POD',color=pal_sel[i],linewidth=6,ax=ax,
                                                linestyle='-',label=i_model)
    #sns.lineplot(data=pr_mean.sort_values('POD'),x='Success Ratio',y='POD',hue='Fold',
     #           sort=False,color='xkcd:purple',ax=ax,linewidth=4,legend=False)
    ax.set_xlim([0,1])
    ax.legend(fontsize=17)
    ax.set_ylim([0,1])
### 11.  make_reliability_diagram:  A function to make a reliability diagram comparing our machine learning models to SHIPS-RII and the SHIPS consensus. 
def make_reliability_diagram(ax,plot_lim,REL_DATA,basin_sel,palette,pct_range,models_skip=None,alpha=1):
    # Inputs:
    # ax: axes for figure
    # plot_lim: limits for reliability diagram (extends beyond 100 b/c we want to add the info about case numbers)
    # REL_DATA: Pandas dataset containing reliability diagram info.  Should include be in format of Basin, Model, 
    #           Predicted Pct, Observed Pct, Observed No. RI, Observed No Total 
    # basin_sel: basin we want to plot (right now, only works for Atlantic and East Pacific)
    # palette: desired color palette (will depend on # of ML models)
    # pct_range: range of predicted RI probabilities
    # models_skip: option argument if we want to skip plotting case numbers for any model 
    # Add 1:1 line
    ax.plot([0,plot_lim],[0,plot_lim],linewidth=3,color='xkcd:slate grey')
    # Add models--first we'll add dots for each predicted probability
    sns.scatterplot(data=REL_DATA.xs(basin_sel).reset_index().sort_values('Model'),x='Predicted Pct',y='Observed Pct',hue='Model',
                palette=sns.set_palette(palette),ax=ax,s=180,alpha=0.9)
    # Then, lines to connect dots
    sns.lineplot(data=REL_DATA.xs(basin_sel).reset_index().sort_values('Model'),x='Predicted Pct',y='Observed Pct',
            hue='Model',palette=sns.set_palette(palette),ax=ax,linewidth=5,legend=False)
    # Formatting
    ax.set_ylim([-0.5,plot_lim])
    ax.set_xlim([-0.5,plot_lim])
    ax.set_xticks([5,10,20,30,40,50,60,70,80,90,100])
    ax.set_yticks([5,10,20,30,40,50,60,70,80,90,100])
    ax.tick_params(axis='y',labelsize=16)
    ax.tick_params(axis='x',labelsize=16)
    ax.legend(fontsize=13,loc='lower right')
    ax.set_xlabel('Predicted RI Probability',fontsize=22)
    ax.set_ylabel('Observed RI Probability',fontsize=22)
    # Calculate total number of observations in each predicted probability bin
    plt_nums = REL_DATA.xs(basin_sel).sort_values(['Predicted Pct']).reset_index().set_index(['Predicted Pct'])
    ax.grid()
    # 
    totals = REL_DATA.xs(basin_sel).reset_index().sort_values('Predicted Pct').set_index(['Model','Predicted Pct'])
    models_list = REL_DATA.reset_index().sort_values(by='Model')['Model'].unique().tolist()
    # Add text containing number of obs per predicted RI probability
    ycount = 0
    for imod in np.arange(0,len(models_list)):
        mod_sel = totals.xs(models_list[imod])
        # print(models_list[imod])
        if models_list[imod] in models_skip:
            ycount = ycount
        else:
            ycount = ycount + 1
        for i_pct in pct_range:
            if i_pct in mod_sel.index:
                ix_mod = mod_sel.xs(i_pct)['Observed No Total'].astype(int)
            else:
                ix_mod = 0
            i_color = sns.color_palette()[imod]
            if models_list[imod] in models_skip:
                ycount = ycount
                # print('do not plot')
            else:
                yval = 101+ycount*3
                ax.text((i_pct-4 if i_pct == 5 else i_pct -2),yval,ix_mod,color=i_color,fontsize=15,weight='semibold')
    return(ax)
#
def _get_pofd_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.

    POFD = probability of false detection
    POD = probability of detection

    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid

    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = int(numpy.ceil(1. / pofd_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))

    unique_pofd_values = numpy.linspace(
        0, 1, num=num_pofd_values + 1, dtype=float
    )
    unique_pofd_values = unique_pofd_values[:-1] + pofd_spacing / 2

    unique_pod_values = numpy.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return numpy.meshgrid(unique_pofd_values, unique_pod_values[::-1])
def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        PEIRCE_SCORE_LEVELS
    ))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object
###
