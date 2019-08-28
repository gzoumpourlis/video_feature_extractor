# video_feature_extraction

Some scripts to extract video features using C3D and ResNet, for the following datasets: Kinetics-100, UCF-101, HMDB51, Olympic Sports.


Here are the steps:

1. Extract dataset's videos to local PC. For example at
```
/home/george/datasets/UCF-101
```

2. (OPTIONAL: rsync to server. You can skip this step.)
For example:
```
rsync -avz --progress /home/george/datasets/UCF-101/ george@abc.efg:/shared/datasets/UCF-101
```

3. Run feature extraction script, with batch mode enabled (otherwise it is buggy, for now, sorry!) and with verbose visibility enabled:
For example:
```
python predict_Kinetics-100.py --batch --verbose
```
Or to run the script using the remote directory, set the corresponding remote flag
```
python predict_Kinetics-100.py --remote --batch --verbose
```

4. After the extraction of the .npy feature files, if the dataset needs to be split into train/val/test sets (which is the case for Kinetics-100 for example), then run the script that performs this split, using also a file where the splits are stored (like splits_Kinetics-100.npy). In this way, a folder will be created for each split, and then the features of a class that belongs to each split, will be copied to that split's folder.
```
python copy_npy_to_splits_Kinetics.py
```
IMPORTANT : Check for possible error in the class named hurling_(sports) because of the parentheses (haven't found out yet why is this happening). If it hasn't been copied correctly in its split's folder, please do this manually.

PS : To setup the modified version of Kinetics (let's call it Kinetics-100), used for few-shot video classification, see also this repo:
https://github.com/gzoumpourlis/FewShotKinetics_csv_lists
