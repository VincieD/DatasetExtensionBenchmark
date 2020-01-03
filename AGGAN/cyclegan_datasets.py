"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'horse2zebra_train': 1334,
    'horse2zebra_test': 140,
    'apple2orange_train': 1019,
    'apple2orange_test': 266,
    'lion2tiger_train': 916,
    'lion2tiger_test': 103,
    'summer2winter_yosemite_train': 1231,
    'summer2winter_yosemite_test': 309,
    'summer2winter_road_256_small_2k_ped_train': 1652,
    'INRIA_Person_Dataset_Train_256_summer_winter': 1139,
    'INRIA_Train_256_summer_winter_with_yt_winter':1139,
    '20191218_INRIA_256_full_summer_winter':2066,
    '20191219_INRIA_256_full_summer_winter':2066,
    'video_sum_win':2939,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'horse2zebra_train': '.jpg',
    'horse2zebra_test': '.jpg',
    'apple2orange_train': '.jpg',
    'apple2orange_test': '.jpg',
    'lion2tiger_train': '.jpg',
    'lion2tiger_test': '.jpg',
    'summer2winter_yosemite_train': '.jpg',
    'summer2winter_yosemite_test': '.jpg',
    'summer2winter_road_256_small_2k_ped_train': '.png',
    'INRIA_Person_Dataset_Train_256_summer_winter': '.png',
    'INRIA_Train_256_summer_winter_with_yt_winter':'.png',
    '20191218_INRIA_256_full_summer_winter':'.png',
    '20191219_INRIA_256_full_summer_winter': '.png',
    'video_sum_win':'.png'
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'horse2zebra_train': './input/horse2zebra/horse2zebra_train.csv',
    'horse2zebra_test': './input/horse2zebra/horse2zebra_test.csv',
    'apple2orange_train': './input/apple2orange/apple2orange_train.csv',
    'apple2orange_test': './input/apple2orange/apple2orange_test.csv',
    'lion2tiger_train': './input/lion2tiger/lion2tiger_train.csv',
    'lion2tiger_test': './input/lion2tiger/lion2tiger_test.csv',
    'summer2winter_yosemite_train': './input/summer2winter_yosemite/summer2winter_yosemite_train.csv',
    'summer2winter_yosemite_test': './input/summer2winter_yosemite/summer2winter_yosemite_test.csv',
    'summer2winter_road_256_small_2k_ped_train': './input/summer2winter_road_256_small_2k_ped/train.csv',
    'INRIA_Person_Dataset_Train_256_summer_winter': './input/INRIA_Person_Dataset_Train_256_summer_winter/test.csv',
    'INRIA_Train_256_summer_winter_with_yt_winter':'./input/INRIA_Train_256_summer_winter_with_yt_winter/train.csv',
    '20191218_INRIA_256_full_summer_winter':'./input/20191218_INRIA_256_full_summer_winter/train.csv',
    '20191219_INRIA_256_full_summer_winter': './input/20191219_INRIA_256_full_summer_winter/train.csv',
    'video_sum_win':'./input/video_sum_win/train.csv'

}
