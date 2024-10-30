#!/usr/bin/env python3

from mayavi import mlab
mlab.options.offscreen = True

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
import sys
import skvideo.io
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from matplotlib.pyplot import get_cmap

from .common import make_process_fun, get_nframes, get_video_name, get_video_params, get_data_length, natural_keys


def connect(points, bps, bp_dict, individuals, nparts, color):
    ixs = [bp_dict[bp] + individuals * nparts for bp in bps]
    # print(ixs)
    return mlab.plot3d(points[ixs, 0], points[ixs, 1], points[ixs, 2],
                       np.ones(len(ixs)), reset_zoom=False,
                       color=color, tube_radius=None, line_width=10)

def connect_all(points, scheme, bp_dict, cmap, individuals, nparts):
    lines = []
    for id in range(len(individuals)):
        for i, bps in enumerate(scheme):
            line = connect(points, bps, bp_dict, id, nparts, color=cmap(i)[:3])
            lines.append(line)
    return lines

def update_line(line, points, bps, bp_dict, individuals, nparts):
    ixs = [bp_dict[bp] + individuals * nparts for bp in bps]
    # ixs = [bodyparts.index(bp) for bp in bps]
    new = np.vstack([points[ixs, 0], points[ixs, 1], points[ixs, 2]]).T
    line.mlab_source.points = new

def update_all_lines(lines, points, scheme, bp_dict, individuals, nparts):
    for id in range(len(individuals)):
        for line, bps in zip(lines, scheme):
            update_line(line, points, bps, bp_dict, id, nparts)



def visualize_labels(config, labels_fname, outname, fps=300):

    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    data = pd.read_csv(labels_fname)
    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))

    individuals = [id.replace('_' + bodyparts[0] + '_error', '') for id in [x for x in cols if '_' + bodyparts[0] + '_error' in x]]

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    id_dict = dict(zip(individuals, range(len(individuals))))

    all_ind_points = []
    all_ind_errors = []
    all_ind_scores = []
    all_ind_ncams = []

    for id in individuals:
        all_points = np.array([np.array(data.loc[:, (id + '_' + bp+'_x', id + '_' + bp+'_y', id + '_' + bp+'_z')])
                               for bp in bodyparts], dtype='float64')
        all_ind_points.append(all_points)

        all_errors = np.array([np.array(data.loc[:, id + '_' + bp+'_error'])
                               for bp in bodyparts], dtype='float64')
        all_ind_errors.append(all_errors)

        all_scores = np.array([np.array(data.loc[:, id + '_' + bp+'_score'])
                               for bp in bodyparts], dtype='float64')
        all_ind_scores.append(all_scores)

        all_ncams = np.array([np.array(data.loc[:, id + '_' + bp+'_ncams'])
                          for bp in bodyparts], dtype='float64')
        all_ind_ncams.append(all_ncams)

    for id, all_errors in enumerate(all_ind_errors):
        if config['triangulation']['optim']:
            all_errors[np.isnan(all_errors)] = 0
        else:
            all_errors[np.isnan(all_errors)] = 10000
        good = (all_errors < 100)
        all_ind_points[id][~good] = np.nan

        not_enough_points = np.mean(all_ind_ncams[id] >= 2, axis=1) < 0.2
        all_ind_points[id][not_enough_points] = np.nan

    # all_points_flat = all_points.reshape(-1, 3)
    all_points_flat = [all_ind_points[id].reshape(-1, 3) for id in range(len(individuals))]
    check = [~np.isnan(all_points_flat[id][:, 0]) for id in range(len(individuals))]

    if np.sum(check) < 10:
        print('too few points to plot, skipping...')
        return
    
    # low, high = np.percentile(all_points_flat[check], [5, 95], axis=0)
    low = [np.percentile(all_points_flat[id][check[id]], 5, axis=0) for id in range(len(individuals))]
    high = [np.percentile(all_points_flat[id][check[id]], 95, axis=0) for id in range(len(individuals))]

    nparts = len(bodyparts)
    framedict = dict(zip(data['fnum'], data.index))

    # writer = skvideo.io.FFmpegWriter(outname, inputdict={
    #     # '-hwaccel': 'auto',
    #     '-framerate': str(fps),
    # }, outputdict={
    #     '-vcodec': 'h264', '-qp': '28', '-pix_fmt': 'yuv420p'
    # })

    size = (500, 500)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    export_video = cv2.VideoWriter(outname, fourcc, fps, size)

    cmap = get_cmap('tab10')


    # points = np.copy(all_points[:, 20])
    # points[0] = low
    # points[1] = high

    points = [np.copy(all_ind_points[id][:, 20]) for id in range(len(individuals))]
    # print(points, len(points))
    for id in range(len(individuals)):
        points[id][0] = low[id]
        points[id][1] = high[id]

    # s = np.arange(points.shape[0])
    # good = ~np.isnan(points[:, 0])

    s = [np.arange(points[id].shape[0]) for id in range(len(individuals))]
    good = [~np.isnan(points[id][:, 0]) for id in range(len(individuals))]

    fig = mlab.figure(bgcolor=(1,1,1), size=(500,500))
    fig.scene.anti_aliasing_frames = 2

    # low, high = np.percentile(points[good, 0], [10,90])
    # scale_factor = (high - low) / 12.0

    low = np.mean([np.percentile(points[id][good[id], 0], 10) for id in range(len(individuals))])
    high = np.mean([np.percentile(points[id][good[id], 0], 90) for id in range(len(individuals))])
    scale_factor = (high - low) / 12.0
    # print(low, high, scale_factor)

    s = [[s[id][bp] + id * nparts for bp in range(nparts)] for id in range(len(individuals))]
    points = np.vstack([points[id] for id in range(len(individuals))])
    s = np.concatenate([s[id] for id in range(len(individuals))])
    # print(points)
    # print(s)

    mlab.clf()
    pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], s,
                        color=(0.8, 0.8, 0.8),
                        scale_mode='none', scale_factor=scale_factor)
    lines = connect_all(points, scheme, bp_dict, cmap, individuals, nparts)
    mlab.orientation_axes()

    view = list(mlab.view())

    mlab.view(focalpoint='auto', distance='auto')

    for framenum in trange(data.shape[0], ncols=70):
        fig.scene.disable_render = True

        if framenum in framedict:
            points = [all_ind_points[id][:, framenum] for id in range(len(individuals))]
            points = np.vstack([points[id] for id in range(len(individuals))])
        else:
            points = np.ones((nparts * len(individuals), 3))*np.nan

        s = np.arange(points.shape[0])
        good = ~np.isnan(points[:, 0])

        new = np.vstack([points[:, 0], points[:, 1], points[:, 2]]).T
        pts.mlab_source.points = new
        update_all_lines(lines, points, scheme, bp_dict, individuals, nparts)

        fig.scene.disable_render = False

        img = mlab.screenshot()

        mlab.view(*view, reset_roll=False)

        export_video.write(img)
        # writer.writeFrame(img)

    export_video.release()
    mlab.close(all=True)
    # writer.close()



def process_session(config, session_path, filtered=False):
    pipeline_videos_raw = config['pipeline']['videos_raw']

    if filtered:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d_filter']
        pipeline_3d = config['pipeline']['pose_3d_filter']
    else:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
        pipeline_3d = config['pipeline']['pose_3d']

    video_ext = config['video_extension']

    vid_fnames = glob(os.path.join(session_path,
                                   pipeline_videos_raw, "*."+video_ext))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        orig_fnames[vidname].append(vid)

    labels_fnames = glob(os.path.join(session_path,
                                      pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)


    outdir = os.path.join(session_path, pipeline_videos_labeled_3d)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.mp4')

        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print(out_fname)

        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)

        visualize_labels(config, fname, out_fname, params['fps'])


label_videos_3d_all = make_process_fun(process_session, filtered=False)
label_videos_3d_filtered_all = make_process_fun(process_session, filtered=True)
