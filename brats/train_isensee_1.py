import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee_1
from unet3d.training import load_old_model, train_model


config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.  128
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])  # =3
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])  # =4
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 2
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped


config["model_file"] = os.path.abspath("isensee_1_model.h5")
config["data_file"] = os.path.abspath("isensee_brats_data.h5")
config["training_file"] = os.path.abspath("isensee_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


def fetch_training_data_files(return_subject_ids=False):
    """
    返回训练文件列表
    """
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    # 若有则加载旧数据集，注意，此时image_shape为之前设置的
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)
    data_file_opened = open_data_file(config["data_file"])

    # 加载/创建模型文件
    from keras.utils.vis_utils import plot_model
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = isensee_1.isensee_1_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                          initial_learning_rate=config["initial_learning_rate"],
                                          n_base_filters=config["n_base_filters"],
                                          depth=4)
        plot_model(model, to_file='isensee_1_uet.png', show_shapes=True)

        model.summary()

    # get training and testing generators
    # ../unet3d/generator.py
    # 创建生成器(generator)，用于后面训练
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    # ../unet3d/training.py
    # 训练一个keras模型
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    # GPU config
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=gpuconfig)
    KTF.set_session(sess)
    main(overwrite=config["overwrite"])
