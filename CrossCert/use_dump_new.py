import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time
from tqdm import tqdm
import joblib

from utils.new import majority_of_mask_single, certified_nowarning_detection, \
    certified_warning_detection, warning_detection, certified_warning_drs, majority_of_drs_single, certified_drs, \
    pc_malicious_label, warning_drs
from utils.pd import one_masking_statistic, double_masking_detection, double_masking_detection_nolemma1
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoints', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='./../../../../public', type=str, help="directory of data")
parser.add_argument('--dataset', default='cifar', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102'), help="dataset")
parser.add_argument("--model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=35, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size", type=int, default=37, help='override dumped file')

t = 100
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

MODEL_NAME = args.model
NUM_IMG = args.num_img

# get model and data loader
# model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
# val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=1, num_img=NUM_IMG, train=False)
#
# device = 'cuda'
# model = model.to(device)
# model.eval()
cudnn.benchmark = True

# generate the mask set
# mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

prediction_map_list_old = joblib.load(os.path.join(DUMP_DIR,
                                                   "prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(66, 66)_s(32, 32)_1000.z"))
# prediction_map_list_mask = joblib.load(os.path.join(DUMP_DIR,
#                                                     "prediction_map_list_mask_predi_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(69, 69)_s(31, 31)_500.z"))
label_list = joblib.load(
    os.path.join(DUMP_DIR, "label_list_imagenet_vit_base_patch16_224_cutout2_128_1000.z"))

prediction_map_list_drs = joblib.load(os.path.join(DUMP_DIR,
                                                   "prediction_map_list_drs_two_mask_imagenet_vit_base_patch16_224_imagenet_drs_37_m37_s1_1000.z"))

ablation_size = args.ablation_size
patch_size = args.patch_size

mask_correct = 0
old_correct = 0
same_correct = 0
robust_same_correct = 0
no_robust_same_correct = 0
robust_sample = 0
total = 0
cannot_certified_warning = 0
certified_warning = 0
cannot_certified_no_warning = 0
certified_no_warning = 0
no_warning_sample = 0
certified_drs_sample = 0

for tor in range(t):
    print("t is " + str(tor))
    for i, (prediction_map_old, label, prediction_map_drs) in enumerate(
            zip(prediction_map_list_old, label_list, prediction_map_list_drs)):
        # init
        total += 1
        certified_result = 0

        # generate a symmetric matrix from a triangle matrix
        prediction_map_old = prediction_map_old + prediction_map_old.T - np.diag(np.diag(prediction_map_old))

        # output label of PC
        output_label_pc = double_masking_precomputed(prediction_map_old)

        # calculate the majority
        major_label_of_ablation = majority_of_drs_single(prediction_map_drs)
        major_label_of_old = majority_of_mask_single(prediction_map_old)

        # whether meet two-masking agreement
        robust = certify_precomputed(prediction_map_old, label)

        drs_label, certified_drs_tag = certified_drs(prediction_map_drs, ablation_size, patch_size)
        # if not robust, we need to certified it should be warned (if the label change?)
        warning_result = warning_drs(prediction_map_drs, output_label_pc, tor, ablation_size, patch_size)
        if not robust:
            malicious_label_list = pc_malicious_label(prediction_map_old, output_label_pc)
            certified_result = certified_warning_drs(malicious_label_list, prediction_map_drs, output_label_pc, tor)
        # if robust, we need to certified it should not be warned if the label not change
        # else:
        #     certified_result = certified_nowarning_detection(prediction_map_mask, output_label_pc,t)
        # warning_result = warning_detection(prediction_map_mask, output_label_pc,t)
        # statistic
        if major_label_of_ablation == label:
            mask_correct += 1
        if major_label_of_old == label:
            old_correct += 1
        if major_label_of_ablation == major_label_of_old and major_label_of_old == label:
            same_correct += 1

        # if output_label_pc == label and robust and warning_result=="no_warning":
        #     robust_sample += 1
        # if certified_result == "cannot_certified_warning" and output_label_pc == label and warning_result=="no_warning":
        #     cannot_certified_warning += 1
        # if certified_result == "certified_warning" and output_label_pc == label and warning_result=="no_warning":
        #     certified_warning += 1
        # if certified_result == "cannot_certified_no_warning" and output_label_pc == label and warning_result=="no_warning":
        #     cannot_certified_no_warning += 1
        # if certified_result == "certified_no_warning" and output_label_pc == label and warning_result=="no_warning":
        #     certified_no_warning += 1
        if output_label_pc == label and robust:
            robust_sample += 1
        if warning_result == "no_warning" and output_label_pc == label:
            no_warning_sample += 1
        if certified_result == "cannot_certified_warning_drs" and output_label_pc == label:
            cannot_certified_warning += 1
        if certified_result == "certified_warning_drs" and output_label_pc == label:
            certified_warning += 1
        # if certified_result == "cannot_certified_no_warning" and output_label_pc == label:
        #     cannot_certified_no_warning += 1
        # if certified_result == "certified_no_warning" and output_label_pc == label:
        #     certified_no_warning += 1
        if certified_drs_tag == "certified_drs" and drs_label == label:
            certified_drs_sample += 1
    print("certified_drs_sample " + str(certified_drs_sample) + ' ' + str(certified_drs_sample / total))
    print("old_correct " + str(old_correct) + ' ' + str(old_correct / total))
    print("mask_correct " + str(mask_correct) + ' ' + str(mask_correct / total))
    print("same_correct " + str(same_correct) + ' ' + str(same_correct / total))
    print("robust_sample " + str(robust_sample) + ' ' + str(robust_sample / total))
    print("certified_drs_sample " + str(certified_drs_sample) + ' ' + str())
    print("no_warning_sample" + str(no_warning_sample) + ' ' + str(no_warning_sample / total))
    print("robust_same_correct " + str(robust_same_correct) + ' ' + str(robust_same_correct / total))
    print("no_robust_same_correct " + str(no_robust_same_correct) + ' ' + str(no_robust_same_correct / total))
    print("cannot_certified_warning " + str(cannot_certified_warning) + ' ' + str(cannot_certified_warning / total))
    print("certified_warning " + str(certified_warning) + ' ' + str(certified_warning / total))
    print("cannot_certified_no_warning " + str(cannot_certified_no_warning) + ' ' + str(
        cannot_certified_no_warning / total))
    print("certified_no_warning " + str(certified_no_warning) + ' ' + str(certified_no_warning / total))
    print("total " + str(total))
    mask_correct = 0
    old_correct = 0
    same_correct = 0
    robust_same_correct = 0
    no_robust_same_correct = 0
    robust_sample = 0
    total = 0
    cannot_certified_warning = 0
    certified_warning = 0
    cannot_certified_no_warning = 0
    certified_no_warning = 0
    no_warning_sample = 0
    certified_drs_sample = 0

# for i, (prediction_map_old, label, prediction_map_mask) in enumerate(
#         zip(prediction_map_list_old, label_list, prediction_map_list_mask)):
#     # init
#     total += 1
#     certified_result = 0
#
#     # generate a symmetric matrix from a triangle matrix
#     prediction_map_old = prediction_map_old + prediction_map_old.T - np.diag(np.diag(prediction_map_old))
#     prediction_map_mask = prediction_map_mask + prediction_map_mask.T - np.diag(np.diag(prediction_map_mask))
#
#     # output label of PC
#     output_label_pc=double_masking_precomputed(prediction_map_old)
#
#     # calculate the majority
#     major_label_of_masks = majority_of_mask_single(prediction_map_mask)
#     major_label_of_old = majority_of_mask_single(prediction_map_old)
#
#     # whether meet two-masking agreement
#     robust = certify_precomputed(prediction_map_old, label)
#
#     # if not robust, we need to certified it should be warned (if the label change?)
#     if not robust:
#         certified_result = certified_warning_detection(prediction_map_old, prediction_map_mask, output_label_pc,t)
#     # if robust, we need to certified it should not be warned if the label not change
#     else:
#         certified_result = certified_nowarning_detection(prediction_map_mask, output_label_pc,t)
#     warning_result = warning_detection(prediction_map_mask, output_label_pc,t)
#     # statistic
#     if major_label_of_masks == label:
#         mask_correct += 1
#     if major_label_of_old == label:
#         old_correct += 1
#     if major_label_of_masks == major_label_of_old and major_label_of_old == label:
#         same_correct += 1
#
#     # if output_label_pc == label and robust and warning_result=="no_warning":
#     #     robust_sample += 1
#     # if certified_result == "cannot_certified_warning" and output_label_pc == label and warning_result=="no_warning":
#     #     cannot_certified_warning += 1
#     # if certified_result == "certified_warning" and output_label_pc == label and warning_result=="no_warning":
#     #     certified_warning += 1
#     # if certified_result == "cannot_certified_no_warning" and output_label_pc == label and warning_result=="no_warning":
#     #     cannot_certified_no_warning += 1
#     # if certified_result == "certified_no_warning" and output_label_pc == label and warning_result=="no_warning":
#     #     certified_no_warning += 1
#     if output_label_pc == label and robust:
#         robust_sample += 1
#     if warning_result == "no_warning" and robust and output_label_pc == label:
#         no_warning_sample += 1
#     if certified_result == "cannot_certified_warning" and output_label_pc == label:
#         cannot_certified_warning += 1
#     if certified_result == "certified_warning" and output_label_pc == label:
#         certified_warning += 1
#     if certified_result == "cannot_certified_no_warning" and output_label_pc == label:
#         cannot_certified_no_warning += 1
#     if certified_result == "certified_no_warning" and output_label_pc == label:
#         certified_no_warning += 1
#
#     print("old_correct " + str(old_correct) + ' ' + str(old_correct / total))
#     print("mask_correct " + str(mask_correct) + ' ' + str(mask_correct / total))
#     print("same_correct " + str(same_correct) + ' ' + str(same_correct / total))
#     print("robust_sample " + str(robust_sample) + ' ' + str(robust_sample / total))
#     print("no_warning_sample"+str(no_warning_sample)+' '+str(no_warning_sample/total))
#     print("robust_same_correct " + str(robust_same_correct) + ' ' + str(robust_same_correct / total))
#     print("no_robust_same_correct " + str(no_robust_same_correct) + ' ' + str(no_robust_same_correct / total))
#     print("cannot_certified_warning " + str(cannot_certified_warning) + ' ' + str(cannot_certified_warning / total))
#     print("certified_warning " + str(certified_warning) + ' ' + str(certified_warning / total))
#     print("cannot_certified_no_warning " + str(cannot_certified_no_warning) + ' ' + str(cannot_certified_no_warning / total))
#     print("certified_no_warning " + str(certified_no_warning) + ' ' + str(certified_no_warning / total))
#     print("total " + str(total))
