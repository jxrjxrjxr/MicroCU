# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import graph_tool.all as gt
import os
# from cv2 import cv2
import cv2
from PIL import Image
from tqdm import tqdm

def draw_alldata(data, data_head, filepath):
    fig = plt.figure(1)
    nrows = int(np.sqrt(data.shape[1])) + 1
    fig.set_size_inches(10 * nrows, 8 * nrows)
    fig.clear()
    axs = fig.subplots(nrows, nrows)
    for i in range(data.shape[1]):
        ax = axs[i//nrows][i%nrows]
        ax.plot(range(data.shape[0]), data[:, i], color='k', alpha=0.8)
#         ax.set_title(str(i + 1) + ":" + data_head[i], fontsize=20)
# 由于service的名字太长，打出来会有重叠，并不美观，因此截取前40个字符
        ax.set_title(str(i + 1) + ":" + data_head[i][:40], fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
    plt.savefig(filepath, dpi=300)


def draw_overlay_histogram(histogram, title, filepath):
    fig = plt.figure(1)
    fig.set_size_inches(5, 3)
    fig.clear()
    # pylint: disable=unsubscriptable-object
    plt.plot(range(np.array(histogram).shape[0]), histogram)
    plt.title(title + "(Sum:{})".format(sum(histogram)))
    plt.savefig(filepath, dpi=400)


def draw_bar_histogram(histogram, auto_threshold_ratio,  title, filepath):
    fig = plt.figure(1)
    fig.set_size_inches(5, 3)
    fig.clear()
    ax = fig.subplots(1, 1)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.2, wspace=0.2, hspace=0.35)
    # pylint: disable=unsubscriptable-object
    ax.bar(range(1, np.array(histogram).shape[0] + 1), histogram, color="k")
    ax.set_xticks(range(1, len(histogram)+1))
    ax.tick_params(axis='x', labelsize=9, labelrotation=60)
    ax.set_xlabel('Services', fontsize='large', horizontalalignment='center')
    ax.axhline(y=auto_threshold_ratio * np.max(histogram), color="r", alpha=0.8, 
                linestyle="--", label=r'$\theta_e * N$')
    ax.legend()
    plt.title(title, fontsize='large')
    plt.savefig(filepath, dpi=400)


def img2video(imgDir, videoDir, fps=30, zn=6, verbose=False, ft='.png'):
    '''
    imgDir: string, image directory
    videoDir: string, video file path
    fps: int, frame per second
    zn: filename related, zn=6->00000*.png
    verbose: boolean, output control
    '''
    num = len([lists for lists in os.listdir(imgDir)])
    img = Image.open(os.path.join(imgDir, str(0).zfill(zn) + ft))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(videoDir, fourcc, fps, img.size)
    for i in range(num):
        im_name = os.path.join(imgDir, str(i).zfill(zn) + ft)
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
        if verbose:
            print(im_name)
    videoWriter.release()
    if verbose:
        print('finish')


def draw_graph_gt(mtx, _head, vers='v1'):
    if type(mtx) == list:
        n = len(_head)
        g = gt.Graph()
        g.add_vertex(n)
        epW = g.new_edge_property("double")
        vpT = g.new_vertex_property("int")
        for i in range(n):
            vpT[g.vertex(i)] = i + 1
        vpS = g.new_vertex_property("string")
        for i in range(n):
            if "dashboard" in _head[i]:
                vpS[g.vertex(i)] = "pentagon"
            elif "events" in _head[i]:
                vpS[g.vertex(i)] = "double_circle"
            else:
                vpS[g.vertex(i)] = "circle"
        for i in range(n):
            for j in range(n):
                if mtx[i][j] > 0:
                    tmpE = g.add_edge(g.vertex(i), g.vertex(j))
                    epW[tmpE] = mtx[i][j]
        deg = g.degree_property_map("total", weight=epW)
        deg.a *= 15
        epW.a *= 5
        pos = g.new_vertex_property("vector<double>")
        for i in range(n):
            angle = (i+0.0)/n*(2*np.pi)
            pos[g.vertex(i)] = [np.cos(angle), np.sin(angle)]
        gt.graph_draw(g, pos=pos, vertex_size=deg, vertex_text=vpT, vertex_shape=vpS, edge_pen_width=epW, bg_color=[255, 255, 255, 1], output="graph.png")
    elif vers == 'v1':
        n = len(_head)
        tLen = mtx.shape[0]
        for t in range(tLen):
            g = gt.Graph()
            g.add_vertex(n)
            epW = g.new_edge_property("double")
            vpT = g.new_vertex_property("int")
            for i in range(n):
                vpT[g.vertex(i)] = i + 1
            vpS = g.new_vertex_property("string")
            for i in range(n):
                if "dashboard" in _head[i]:
                    vpS[g.vertex(i)] = "pentagon"
                elif "events" in _head[i]:
                    vpS[g.vertex(i)] = "double_circle"
                elif "console" in _head[i]:
                    vpS[g.vertex(i)] = "square"
                else:
                    vpS[g.vertex(i)] = "circle"
            for i in range(n):
                for j in range(n):
                    if mtx[t, i, j] > 0.1:
                        tmpE = g.add_edge(g.vertex(i), g.vertex(j))
                        epW[tmpE] = mtx[t, i, j]
            degIn = g.degree_property_map("in", weight=epW)
            degOut = g.degree_property_map("out", weight=epW)
            degOut.a *= 60
            degIn.a /= degIn.a.max()
            epW.a *= 10
            pos = g.new_vertex_property("vector<double>")
            col = g.new_vertex_property("vector<double>")
            for i in range(n):
                angle = (i+0.0)/n*(2*np.pi)
                pos[g.vertex(i)] = [np.cos(angle), np.sin(angle)]
                col[g.vertex(i)] = [degIn[i], 0, 0, 0.8]
            ft = '.svg'
            im_name = os.path.join('.', 'imgFolder', str(t).zfill(3) + ft)
            gt.graph_draw(g, pos=pos,
                             output_size=(800, 800),
                             vertex_size=degOut,
                             vertex_text=vpT,
                             vertex_shape=vpS,
                             vertex_fill_color=col,
                             edge_pen_width=epW,
                             edge_marker_size=8,
                             bg_color=[255, 255, 255, 1],
                             output=im_name)
        img2video('./imgFolder', './imgFolder/test.avi', fps=30, zn=3, ft=ft)
    else:
        n = len(_head)
#         event = [1, 6, 12, 13, 28, 30, 31, 33]
#         xforce = [2, 5, 8, 15, 17, 20, 21, 22]
#         other = [3, 7, 9, 10, 19, 24, 27, 32]
#         test = [4, 11, 14, 26]
#         prod = [16, 18, 23, 25, 29]
#         dash = test + prod
        frontend = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        middle = [13, 14, 15, 16, 17, 18, 19, 20]
        backend = [21, 22, 23, 24, 25]
#         ins = [31, 19, 3, 8, 22, 21, 16]
#         ins = [19, 3, 8, 22, 21, 16]
        ins = [5, 14, 18, 23]
        ous = [u for u in range(1, n + 1) if u not in ins]
        tLen = mtx.shape[0]
        for t in range(tLen):
            g = gt.Graph()
            g.add_vertex(n)
            epW = g.new_edge_property("double")
            vpT = g.new_vertex_property("int")
            for i in range(n):
                vpT[g.vertex(i)] = i + 1
            vpS = g.new_vertex_property("string")
            for i in range(n):
                u = i + 1
#                 if u in event:
#                     vpS[g.vertex(i)] = "double_circle"
#                 elif u in xforce:
#                     vpS[g.vertex(i)] = "square"
#                 elif u in other:
#                     vpS[g.vertex(i)] = "circle"
#                 else:
#                     vpS[g.vertex(i)] = "pentagon"
                if u in frontend:
                    vpS[g.vertex(i)] = "double_circle"
                elif u in middle:
                    vpS[g.vertex(i)] = "square"
                elif u in backend:
                    vpS[g.vertex(i)] = "circle"
                else:
                    vpS[g.vertex(i)] = "pentagon"
            for i in range(n):
                for j in range(n):
                    if mtx[t, i, j] > 0.4:
                        tmpE = g.add_edge(g.vertex(i), g.vertex(j))
                        epW[tmpE] = mtx[t, i, j]
            degIn = g.degree_property_map("in", weight=epW)
            degOut = g.degree_property_map("out", weight=epW)
            # degOut.a  = 90 * (np.sqrt(degOut.a) * 0.5 + 0.4)
            vpFS = g.new_vertex_property("int")
            vpTC = g.new_vertex_property("vector<double>")
            for u in g.vertices():
                if vpT[u] in ins:
                    degOut[u]  = 120 * (np.sqrt(degOut[u]) * 0.5 + 0.4)
                    vpFS[u] = 40
                    vpTC[u] = [0, 255, 255, 1]
                else:
                    degOut[u]  = 90 * (np.sqrt(degOut[u]) * 0.5 + 0.4)
                    vpFS[u] = 12
                    vpTC[u] = [255, 255, 255, 1]
            # degOut.a *= 60
            degIn.a /= degIn.a.max()
            # epW.a *= 10
            # epW.a /= epW.a.max() / 30
            if epW.a.size > 0:
                for e in g.edges():
                    if vpT[e.source()] in ins and vpT[e.target()] in ins:
                        epW[e] = np.sqrt(epW[e]) * 30
                    else:
                        epW[e] = np.sqrt(epW[e]) * 10
            pos = g.new_vertex_property("vector<double>")
            col = g.new_vertex_property("vector<double>")

            # get layout start
            # for i, u in enumerate(event):
            #     angle = i / len(event) * (2 * np.pi)
            #     pos[g.vertex(u - 1)] = (np.cos(angle), np.sin(angle) - 3)
            # for i, u in enumerate(xforce):
            #     angle = i / len(xforce) * (2 * np.pi)
            #     pos[g.vertex(u - 1)] = (np.cos(angle) * 0.7 - 3, np.sin(angle) * 0.7)
            # for i, u in enumerate(other):
            #     angle = i / len(other) * (2 * np.pi)
            #     pos[g.vertex(u - 1)] = (np.cos(angle) * 0.7 + 3, np.sin(angle) * 0.7)
            # for i, u in enumerate(test):
            #     angle = (6 + i + 1) / 12 * np.pi
            #     pos[g.vertex(u - 1)] = (np.cos(angle) * 3, np.sin(angle) * 3 + 1)
            # for i, u in enumerate(prod):
            #     angle = (6 - i) / 12 * np.pi
            #     pos[g.vertex(u - 1)] = (np.cos(angle) * 3, np.sin(angle) * 3 + 1)

            for i in range(len(ins)):
                angle  = i / len(ins) * 2 * np.pi
                pos[g.vertex(ins[i] - 1)] = (np.cos(angle), np.sin(angle))
            for i in range(len(ous)):
                angle = i / len(ous) * 2 * np.pi
                pos[g.vertex(ous[i] - 1)] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))
            # get layout end

            # for i in range(n):
            #     col[g.vertex(i)] = [degIn[i], 0, 0, 0.8]

            for i in range(len(ins)):
                col[g.vertex(ins[i] - 1)] = [degIn[ins[i] - 1], 0, 0, 0.9]
            for i in range(len(ous)):
                col[g.vertex(ous[i] - 1)] = [degIn[ous[i] - 1], 0, 0, 0.2]
            ecol = g.new_edge_property("vector<double>")
            for e in g.edges():
                if vpT[e.source()] in ins and vpT[e.target()] in ins:
                    ecol[e] = [0, 0, 0, 1]
                else:
                    ecol[e] = [0, 0, 0, 0.3]

            # inter = g.new_edge_property("bool")
            # for e in g.edges():
            #     inter[e] = True
            #     if vpT[e.target()] in event:
            #         inter[e] = False
            #     elif vpT[e.target()] in xforce + other and vpT[e.source()] in dash:
            #         inter[e] = False
            #     elif vpT[e.target()] in dash and vpT[e.source()] in dash:
            #         inter[e] = False
            # g.set_edge_filter(inter)
            inter = g.new_edge_property("bool")
            for e in g.edges():
                inter[e] = True
                x, y = vpT[e.source()], vpT[e.target()]
                if x in ins and y in ins:
                    if ins.index(x) + 1 != ins.index(y):
                        inter[e] = False
            g.set_edge_filter(inter)

            ft = '.png'
            im_name = os.path.join('.', 'imgFolder-0526', str(t).zfill(3) + ft)
            epMS = g.new_edge_property("double")
            epMS.a = 2 * epW.a
            gt.graph_draw(g, pos=pos,
                             output_size=(1200, 1200),
                             vertex_size=degOut,
                             vertex_text=vpT,
                             vertex_shape=vpS,
                             vertex_fill_color=col,
                             edge_pen_width=epW,
                             edge_marker_size=epMS,
                             bg_color=[255, 255, 255, 1],
                             edge_color=ecol,
                             vertex_text_color=vpTC,
                             vertex_font_size=vpFS,
                             output=im_name)
#             break
#         return
        img2video('./imgFolder-0526', './imgFolder-0526/fault_prop8th.avi', fps=6, zn=3, ft=ft)


mtx = np.random.random((100, 33, 33))
_head = ["dashboard" for i in range(33)]
draw_graph_gt(mtx, _head, 'v2')

import os
os.getcwd()

os.listdir("./")

mtx = np.load("./data/dy_mat.npy")

mtx.shape

for i in range(200):
    mtx[i, :, :] = (mtx[i, :, :] - mtx[i, :, :].min()) / (mtx[i, :, :].max() - mtx[i, :, :].min())

_head = [
    "social-follow-user",
    "social-recommender",
    "social-unique-id",
    "social-url-shorten",
    "social-video",
    "social-image",
    "social-text",
    "social-user-tag",
    "social-favorite",
    "social-search",
    "social-ads",
    "social-read-post",
    "social-login",
    "social-compose-post",
    "social-blocked-users",
    "social-read-timeline",
    "social-user-info",
    "social-posts-storage",
    "social-write-timeline",
    "social-write-graph",
    "social-read-timeline-db",
    "social-user-info-db",
    "social-posts-storage-db",
    "social-write-timeline-db",
    "social-write-graph-db"]
draw_graph_gt(mtx, _head, 'v2')


