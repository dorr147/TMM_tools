import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['kaiti']
mpl.rcParams['axes.unicode_minus'] = False

def plot_1d_curves_respectly(x, y,title=None, label=None, xlim=None, ylim=None,
                             legend=False, xlabel=None, ylabel=None, savepath=None,
                             color=None):
    '''x is a list include all curve parameter'''
    for i,x_plot in enumerate(x):
        title_plot=f"{title[i]}" if title is not None else f"{i}"
        y_plot=y[i]
        label_plot=f"{label[i]}" if label is not None else None
        xlim_plot=xlim[i] if xlim is not None else None
        ylim_plot=ylim[i] if ylim is not None else None
        xlabel_plot=f"{xlabel[i]}" if xlabel is not None else None
        ylabel_plot=f"{ylabel[i]}" if ylabel is not None else None
        color_plot=f"{color[i]}" if color is not None else None
        plt.figure(title_plot)
        plt.plot(x_plot, y_plot, label=label_plot, color=color_plot)
        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            save_path_plot = f"{savepath}/{title_plot}.png"
            plt.savefig(save_path_plot)
        plt.xlabel(xlabel_plot)
        plt.ylabel(ylabel_plot)
        plt.title(title_plot)
        if legend==True:
            plt.legend()
        plt.xlim(xlim_plot)
        plt.ylim(ylim_plot)
    plt.show()

def plot_1d_curves_in_same_figure(x, ylis, title=None, xlabel=None, ylabel=None, titlefontsize=16, xlabelfontsize=14, ylabelfontsize=14,
                                  xlim=None, ylim=None,
                                  colors=None, linestyles=None, linewidths=None, markers=None, markersize=None,
                                  legend=False, legend_loc="best", grid=False, labels=None,
                                  figsize=(8,7), dpi=100,
                                  savepdir=None, isshow=True):
    title = title if title is not None else None
    xlim = xlim if xlim is not None else None
    ylim = ylim if ylim is not None else None
    xlabel=xlabel if xlabel is not None else None
    ylabel=ylabel if ylabel is not None else None
    linestyles=linestyles if linestyles is not None else None
    linewidths=linewidths if linewidths is not None else None
    markers=markers if markers is not None else None


    fig = plt.figure(title,figsize=figsize,dpi=dpi)
    plt.title(title,fontsize=titlefontsize)
    plt.xlabel(xlabel,fontsize=xlabelfontsize)
    plt.ylabel(ylabel,fontsize=ylabelfontsize)
    for i,y in enumerate(ylis):
        label_plot = labels[i] if labels is not None else None
        color_plot = colors[i] if colors is not None else None
        linestyle_plot = linestyles[i] if linestyles is not None else None
        marker_plot = markers[i] if markers is not None else None
        linewidth_plot = linewidths[i] if linewidths is not None else None
        plt.plot(x,y,label=label_plot,color=color_plot,ls=linestyle_plot,lw=linewidth_plot,marker=marker_plot,markersize=markersize)
    plt.grid(grid)
    if legend == True:
        plt.legend(loc=legend_loc)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if savepdir is not None:
        os.makedirs(savepdir, exist_ok=True)
        savepng=f"{savepdir}/{title}.png"
        plt.savefig(savepng)
    if isshow:
        plt.show()
    else:
        return fig


def plot_heatmap(x, y, Z, dpi=100, figsize=(8, 7), cmp="plasma", xticknum=6, yticknum=6,
                 barticknum=6, xlabel="x", ylabel="y", title="heatmap", accuracy=2,
                 xylabelfontsize=14, titlefontsize=18,rotation=0,xytickfontsize=10,bartickfontsize=8,savedir=None,isshow=True):
    f = plt.figure(dpi=dpi, figsize=figsize)
    ax = f.subplots()
    im = ax.imshow(Z, cmap=cmp, aspect='auto',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    xtick_positions = np.linspace(x.min(), x.max(), xticknum)
    ytick_positions = np.linspace(y.min(), y.max(), yticknum)
    xtick_labels = [f"{val:.{accuracy}f}" for val in xtick_positions]
    ytick_labels = [f"{val:.{accuracy}f}" for val in ytick_positions]

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels,fontsize=xytickfontsize,rotation=rotation)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels,fontsize=xytickfontsize, rotation=rotation)
    ax.set_xlabel(xlabel, fontsize=xylabelfontsize)
    ax.set_ylabel(ylabel, fontsize=xylabelfontsize)
    ax.set_title(title, fontsize=titlefontsize)

    bar = plt.colorbar(im)
    bartick_positions = np.linspace(Z.min(), Z.max(), barticknum)
    bartick_labels = [f"{val:.{accuracy}f}" for val in bartick_positions]
    bar.set_ticks(bartick_positions)
    bar.set_ticklabels(bartick_labels,fontsize=bartickfontsize,rotation=rotation)
    bar.set_label("colorbar")

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        savepng=f"{savedir}/{title}.png"
        plt.savefig(savepng)

    plt.tight_layout()
    if isshow:
        plt.show()
        plt.close()



if __name__ == '__main__':
    # x = np.linspace(7, 10, 100)
    # y = np.linspace(0, 1, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = X ** 2 + Y ** 2
    # plot_heatmap(x, y, Z, cmp="coolwarm",xticknum=9, yticknum=6, barticknum=9, xlabel="xdata", ylabel="ydata", title="Z", rotation=0,savedir="11")
    x=np.linspace(-np.pi,np.pi,100)
    y1=np.sin(x-0.5*np.pi)
    y2=np.sin(x)
    y3=np.sin(x+0.5*np.pi)
    ylis=[y1,y2,y3]
    plot_1d_curves_in_same_figure(x, ylis, title="sin-cos", xlabel="x", ylabel="y",
                                  xlim=(-np.pi,np.pi), ylim=(-1,1), linestyles=["-","--",":"],
                                  colors=["r","g","b"], linewidths=[1.5,2,2.5], markers=["o","^","_"], markersize=3,
                                  labels=["y1","y2","y3"], legend=True,grid=True,legend_loc="upper right")