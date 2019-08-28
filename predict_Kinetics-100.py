import argparse
import numpy as np
import torch
from torch.autograd import Variable
import os
from glob import glob
from C3D_model import C3D
import cv2
from torchvision import models

def preprocess_img(img, size, batch):

    img = cv2.resize(img, (size, size))
    img = img.transpose(2, 0, 1)
    if batch == False:
        img = np.expand_dims(img, axis=0)
    img = np.float32(img)

    return img

def preprocess_clip(clip, mean_cube, batch):
    
    clip = clip.transpose(3, 0, 1, 2)
    clip -= mean_cube
    clip = clip[:, :, 8:120, 30:142]
    if batch==False:
        clip = np.expand_dims(clip, axis=0)
    clip = np.float32(clip)
    if batch == False:
        clip = torch.from_numpy(clip)
    
    return clip

def get_np_clip(clip_path, verbose=True):
    
    cap = cv2.VideoCapture(clip_path)
    vid = []
    vid_raw = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid_raw.append(img)
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)
    vid_raw = np.array(vid_raw, dtype=np.float32)

    return vid, vid_raw

def read_labels_from_file(filepath):
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser("C3D & ResNet feature extraction")
    parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument('--videos_root_local', type=str, default='/home/george/datasets/ActivityNet/Crawler/Kinetics/dataset', help="set videos root path")
    parser.add_argument('--videos_root_remote', type=str, default='/shared/datasets/Kinetics_100', help="set videos root path")
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--c3d_model_root', type=str, default='model', help="set C3D model root path")
    parser.add_argument('--video_list', type=str, default='video_Kinetics-100.list', help="set video list path")
    parser.add_argument('--preds_c3d_root', type=str, default='preds_c3d_Kinetics-100', help="set video C3D predictions path, to store .npy files")
    parser.add_argument('--preds_cnn_root', type=str, default='preds_cnn_Kinetics-100', help="set video CNN predictions path, to store .npy files")
    parser.add_argument('--c3d_batch_size', type=int, default=6, help="set C3D batch size")
    parser.add_argument('--cnn_batch_size', type=int, default=32, help="set CNN batch size")
    parser.add_argument('--batch', action='store_true')
    parser.set_defaults(batch=True)
    parser.add_argument('--gpu', type=int, default=0, help="set gpu id")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    
    if args.batch==False:
        print("Currently, you *have* to run this in batch mode, i.e. batch_size>1. Quitting...")
        quit()
    
    # if args.cuda:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #     if torch.cuda.is_available():
    #         print('Using CUDA device {}'.format(args.gpu))

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.remote:
        args.videos_root = args.videos_root_remote
    else:
        args.videos_root = args.videos_root_local

    args.c3d_model_root = os.path.join(os.getcwd(), args.c3d_model_root)
    args.video_list = os.path.join(os.getcwd(), args.video_list)
    args.preds_c3d_root = os.path.join(os.getcwd(), args.preds_c3d_root)

    model_path = os.path.join(args.c3d_model_root, 'c3d.pickle')
    mean_path = os.path.join(args.c3d_model_root, 'c3d_mean.npy')
    labels = read_labels_from_file('labels_Sports-1M.txt')

    ############################################
    # Load ResNet-50

    resnet50_full = models.resnet50(pretrained=True)

    class ResNet50_FC(torch.nn.Module):
        def __init__(self):
            super(ResNet50_FC, self).__init__()
            self.features = torch.nn.Sequential(
                # stop at FC, to extract FC features, not class scores
                *list(resnet50_full.children())[:-1]
            )

        def forward(self, x):
            x = self.features(x)
            return x

    resnet50 = ResNet50_FC()
    if args.cuda==True:
        resnet50 = resnet50.cuda()
    resnet50.eval()

    ############################################

    if not os.path.exists(args.c3d_model_root):
        os.mkdir(args.c3d_model_root)

    if not os.path.exists(model_path):
        model_url = 'http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle'
        download_model_cmd = 'wget {} --directory-prefix {}'.format(model_url, args.c3d_model_root)
        os.system(download_model_cmd)
    if not os.path.exists(mean_path):
        mean_url = 'https://github.com/albertomontesg/keras-model-zoo/raw/master/kerasmodelzoo/data/c3d_mean.npy'
        download_mean_cmd = 'wget {} --directory-prefix {}'.format(mean_url, args.c3d_model_root)
        os.system(download_mean_cmd)

    if not os.path.exists(args.preds_c3d_root):
        os.mkdir(args.preds_c3d_root)
    if not os.path.exists(args.preds_cnn_root):
        os.mkdir(args.preds_cnn_root)
    
    C3D_STEP = 16
    cnn_size = 224
    mean_cube = np.load(mean_path)
    mean_cube = mean_cube[0]
    cnn_mean = np.array((0.485, 0.456, 0.406))
    cnn_std = np.array((0.229, 0.224, 0.225))
    cursor_up = "\x1b[1A"
    erase_line = "\x1b[1A"
    
    net = C3D()
    net.load_state_dict(torch.load(model_path))
    if args.cuda==True:
        net = net.cuda()
    net.eval()

    if args.verbose:
        print('Reading video list')
        print('')
        print('')

    video_paths = []
    file = open(args.video_list, 'r')
    for line in file:
        line = line.rstrip('\n')
        if args.videos_root!='':
            video_path = os.path.join(args.videos_root, line)
        else:
            video_path = line
        video_paths.append(video_path)
    
    vid_cnt = 0
    N_vids = len(video_paths)
    for clip_path in video_paths:
        print(cursor_up + erase_line)
        video_name_with_ext = clip_path.split('/')[-1]
        video_name = video_name_with_ext.split('.')[0]
        preds_filename = video_name + '.npy'
        class_name = clip_path.split('/')[-2]
        class_name = class_name.replace(' ', '_')

        class_preds_c3d_folder = os.path.join(args.preds_c3d_root, class_name)
        if not os.path.exists(class_preds_c3d_folder):
            os.mkdir(class_preds_c3d_folder)
        class_preds_cnn_folder = os.path.join(args.preds_cnn_root, class_name)
        if not os.path.exists(class_preds_cnn_folder):
            os.mkdir(class_preds_cnn_folder)

        c3d_video_preds_path = os.path.join(class_preds_c3d_folder, preds_filename)
        cnn_video_preds_path = os.path.join(class_preds_cnn_folder, preds_filename)

        vid_cnt += 1
        # In case that you're having problems with a specific video file, use something like this
        # if vid_cnt==3021:
        #     continue

        print('{:04d}/{:04d} Processing video "{}"'.format(vid_cnt, len(video_paths), video_name))
        if os.path.exists(c3d_video_preds_path) and os.path.exists(cnn_video_preds_path):
            continue
        print(' ')

        clip_full, clip_full_raw = get_np_clip(clip_path)
        N_frames = clip_full.shape[0]

        #########################################
        # C3D feature extraction

        if not os.path.exists(c3d_video_preds_path):
            N_iters = int(np.ceil(N_frames/C3D_STEP))
            features = []
            frames_t = []
            if args.batch:
                batch_cnt = 0
                batch_clips = []
            for t in range(0, N_iters):
                if t < (N_iters - 1):
                    start_frame = t * C3D_STEP
                else:
                    start_frame = N_frames - C3D_STEP
                batch_c3d_condition = ((N_iters + batch_cnt - t) >= args.c3d_batch_size)
                clip = clip_full[start_frame:(start_frame + C3D_STEP), :, :, :].copy()
                clip = preprocess_clip(clip, mean_cube, args.batch and batch_c3d_condition)
                if args.verbose:
                    print(cursor_up + erase_line)
                    print('Video {:07d}/{:07d} Frame {:07d}/{:07d} | {:02d}% | Using C3D for video "{}" | Batch: {}'.format(
                            vid_cnt, N_vids, start_frame + 1, N_frames, int(100 * (start_frame / N_frames)), video_name, args.batch))
                frames_t.append(start_frame)
                if (not args.batch) or (not batch_c3d_condition):
                    #print('Gathering single clip')
                    with torch.no_grad():
                        if args.cuda:
                            X = Variable(clip.cuda())
                        probs, feats = net(X)
                        feats_cpu = feats.data.cpu().numpy()
                    features.append(feats_cpu[0])
                elif batch_c3d_condition:
                    batch_cnt += 1
                    batch_clips.append(clip)
                    #print('Gathering video batch {}/{}'.format(batch_cnt, args.c3d_batch_size))
                    if batch_cnt == args.c3d_batch_size:
                        clip = np.array(batch_clips)
                        clip = torch.from_numpy(clip)
                        with torch.no_grad():
                            X = Variable(clip)
                            if args.cuda:
                                X = X.cuda()
                            probs, feats = net(X)
                            feats_cpu = feats.data.cpu().numpy()
                        batch_clips = []
                        for batch_iter in range(0,args.c3d_batch_size):
                            features.append(feats_cpu[batch_iter])
                        batch_cnt = 0
                clip = []
                X = []
            assert(len(features)==len(frames_t))
            #print('C3D : gathered %d vectors in %d times' % ( len(features), len(frames_t) ))
            video_dict_c3d = {'features' : features,
                              'frames_t' : frames_t}
            np.save(c3d_video_preds_path, video_dict_c3d)

        #########################################
        # CNN feature extraction

        if not os.path.exists(cnn_video_preds_path):
            if args.batch:
                batch_cnt = 0
                batch_imgs = []
            features = []
            frames_t = []
            for t in range(0, N_frames):
                frame_index = t
                batch_cnn_condition = ((N_frames + batch_cnt - t) >= args.cnn_batch_size)
                img = clip_full_raw[frame_index].copy()
                if args.verbose:
                    print(cursor_up + erase_line)
                    print(
                        'Video {:07d}/{:07d} Frame {:07d}/{:07d} | {:02d}% | Using ResNet for video "{}"'.format(
                            vid_cnt, N_vids, frame_index + 1, N_frames, int(100 * (frame_index / N_frames)),
                            video_name))
                img = preprocess_img(img, cnn_size, args.batch and batch_cnn_condition)
                frames_t.append(frame_index)
                if (not args.batch) or (not batch_cnn_condition):
                    #print('Gathering single image')
                    img = img / 255.0
                    for ch_i in range(0, 3):
                        img[0, ch_i, :, :] = img[0, ch_i, :, :] - cnn_mean[ch_i]
                        img[0, ch_i, :, :] = img[0, ch_i, :, :] / cnn_std[ch_i]
                    img = torch.from_numpy(img)
                    with torch.no_grad():
                        X = Variable(img)
                        if args.cuda:
                            X = X.cuda()
                        feats = resnet50(X)
                        feats_cpu = feats.data.cpu().numpy()
                    features.append(feats_cpu[0].flatten())
                elif batch_cnn_condition:
                    batch_cnt += 1
                    img = img / 255.0
                    for ch_i in range(0,3):
                        img[ch_i,:,:] = img[ch_i,:,:] - cnn_mean[ch_i]
                        img[ch_i, :, :] = img[ch_i, :, :] / cnn_std[ch_i]
                    batch_imgs.append(img)
                    #print('Gathering image batch {}/{}'.format(batch_cnt, args.cnn_batch_size))
                    if batch_cnt == args.cnn_batch_size:
                        img = np.array(batch_imgs)
                        img = torch.from_numpy(img)
                        with torch.no_grad():
                            X = Variable(img)
                            if args.cuda:
                                X = X.cuda()
                            feats = resnet50(X)
                            feats_cpu = feats.data.cpu().numpy()
                        batch_imgs = []
                        for batch_iter in range(0, args.cnn_batch_size):
                            features.append(feats_cpu[batch_iter].flatten())
                        batch_cnt = 0
                img = []
                X = []
            assert(len(features)==len(frames_t))
            #print('CNN : gathered %d vectors in %d times' % ( len(features), len(frames_t) ))
            video_dict_cnn = {'features': features,
                              'frames_t': frames_t}
            np.save(cnn_video_preds_path, video_dict_cnn)

        #########################################

if __name__ == '__main__':
    main()
