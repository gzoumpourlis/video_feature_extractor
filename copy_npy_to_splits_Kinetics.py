import os
import numpy as np

# setting this to False, will not copy the files. instead, you'll just see printed messages of where the files would be copied
dry_run = False

root_path = os.getcwd()
preds_c3d_all_path = 'preds_c3d_Kinetics-100'
preds_cnn_all_path = 'preds_cnn_Kinetics-100'
preds_c3d_split_path = 'preds_c3d_split_Kinetics-100'
preds_cnn_split_path = 'preds_cnn_split_Kinetics-100'

CMN_splits = 'splits_Kinetics-100.npy'

splits_raw = np.load(CMN_splits)
splits = np.ndarray.tolist(splits_raw)

for set_name in ['train', 'val', 'test']:
    classes_set = splits[set_name]
    cnt_class = 0
    for class_name in classes_set:
        cnt_class += 1
        class_name = class_name.replace(' ', '_')
        source_path_c3d = os.path.join(root_path, preds_c3d_all_path, class_name)
        target_path_c3d = os.path.join(root_path, preds_c3d_split_path, set_name)
        source_path_cnn = os.path.join(root_path, preds_cnn_all_path, class_name)
        target_path_cnn = os.path.join(root_path, preds_cnn_split_path, set_name)
        message = '{} set \t| C3D/CNN | Copying class {:03d}/{:03d} : "{}"'.format(set_name, cnt_class, len(classes_set), class_name)
        print(message)
        copy_c3d_cmd = 'cp -r {} {}'.format(source_path_c3d, target_path_c3d)
        copy_cnn_cmd = 'cp -r {} {}'.format(source_path_cnn, target_path_cnn)
        if dry_run==True:
            print(copy_c3d_cmd)
            print(copy_cnn_cmd)
        else:
            os.system(copy_c3d_cmd)
            os.system(copy_cnn_cmd)
