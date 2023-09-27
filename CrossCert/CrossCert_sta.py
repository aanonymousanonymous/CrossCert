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
    pc_malicious_label, warning_drs, double_masking_precomputed_with_case_num, warning_analysis, malicious_list_drs, \
    malicious_list_compare, pc_malicious_label_with_location, mask_ablation_for_all, \
    suspect_column_list_cal, certified_with_location, check_maskfree_empty, suspect_column_list_cal_fix, \
    pc_malicious_label_check, double_masking_precomputed_with_case_num_modify, warning_analysis_modify
from utils.pd import one_masking_statistic, double_masking_detection, double_masking_detection_nolemma1
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoints', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='./../../../../public', type=str, help="directory of data")
parser.add_argument('--dataset', default='cifar100', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102'), help="dataset")
# parser.add_argument("--pc_model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
# parser.add_argument("--drs_model", default='vit_base_patch16_224', type=str, help="model name")
parser.add_argument("--pc_model", default='resnetv2_50x1_bit_distilled_cutout2_128', type=str, help="model name")
parser.add_argument("--drs_model", default='resnetv2_50x1_bit_distilled', type=str, help="model name")

parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=32, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size", type=int, default=37, help='override dumped file')
parser.add_argument("--modify", type=bool, default=True, help='override dumped file')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_args()
print(args)
print(args.patch_size)
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
NUM_IMG = args.num_img
PC_MODEL_NAME = args.pc_model
MODEL_NAME = PC_MODEL_NAME
DRS_MODEL_NAME = args.drs_model
ablation_size = args.ablation_size
patch_size = args.patch_size
modify = args.modify
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=16, num_img=NUM_IMG, train=False)

device = 'cuda'
# model = model.to(device)
model.eval()
cudnn.benchmark = True

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

if not args.override and os.path.exists(
        os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z')):
    print('loading two-mask predictions')
    # maskfree_all_list = joblib.load(os.path.join(DUMP_DIR, 'maskfree_all_list'+str(args.num_mask)+'_'+str(patch_size)+'.z'))
    # suspect_column_list= joblib.load(os.path.join(DUMP_DIR, 'suspect_column_list'+str(args.num_mask)+'_'+str(patch_size)+'.z'))
    suspect_column_list = joblib.load(
        os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
else:
    # maskfree_all_list = mask_ablation_for_all(args.num_mask, mask_list)
    # suspect_column_list = suspect_column_list_cal(maskfree_all_list)
    suspect_column_list = suspect_column_list_cal_fix(mask_list)
    joblib.dump(suspect_column_list,
                os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
    # joblib.dump(maskfree_all_list,
    #             os.path.join(DUMP_DIR, 'maskfree_all_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
    # joblib.dump(suspect_column_list,
    #             os.path.join(DUMP_DIR, 'suspect_column_list' + str(args.num_mask) + '_' + str(patch_size) + '.z'))
# print("check_maskfree_empty(maskfree_all_list)")
# print(check_maskfree_empty(suspect_column_list))
prediction_map_list_pc = joblib.load(os.path.join(DUMP_DIR,
                                                  "prediction_map_list_two_mask_{}_{}_m{}_s{}_{}.z".format(DATASET,
                                                                                                           PC_MODEL_NAME,
                                                                                                           str(MASK_SIZE),
                                                                                                           str(MASK_STRIDE),
                                                                                                           NUM_IMG)))
label_list = joblib.load(
    os.path.join(DUMP_DIR, "label_list_{}_{}_{}.z".format(DATASET, PC_MODEL_NAME, NUM_IMG)))

prediction_map_list_drs = joblib.load(os.path.join(DUMP_DIR,
                                                   "prediction_map_list_drs_two_mask_{}_{}_{}_drs_{}_m{}_s1_{}.z".format(
                                                       DATASET, DRS_MODEL_NAME, DATASET, ablation_size, ablation_size,
                                                       NUM_IMG)))


# prediction_map_list_drs_two_mask_imagenet_vit_base_patch16_224_imagenet_drs_37_m37_s1_50000.z

def static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    # if output_label_pc == output_label_drs and (robust_pc or robust_drs):
    #     return True
    # # elif robust_pc:
    # #     return True
    # if output_label_pc == output_label_drs and (robust_pc or robust_drs):
    #     return True
    # elif robust_pc:
    #     return True
    if robust_pc or (output_label_pc == output_label_drs and robust_drs):
        return True
    return False


def static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    if output_label_pc == output_label_drs and (robust_pc and robust_drs):
        return True
    return False


# clean_sample_warning = 0
# clean_sample_nowarning = 0
# sta_certified_sample = 0
# label_certified_sample = 0
# not_sta_certified_sample = 0
# not_label_certified_sample = 0
# stable_sample = 0
# cert_drs = 0
# OMA_detection_sample=0
# pc_recovery_cert_sample=0
# location_certified_sample=0
# for i, (prediction_map_pc, label, prediction_map_drs) in enumerate(
#         zip(prediction_map_list_pc, label_list, prediction_map_list_drs)):
#     # init
#     # generate a symmetric matrix from a triangle matrix
#     prediction_map_pc = prediction_map_pc + prediction_map_pc.T - np.diag(np.diag(prediction_map_pc))
#
#     # output label of PC
#     output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
#     # calculate the majority
#     major_label_of_ablation = majority_of_drs_single(prediction_map_drs)
#     # warning analysis
#     warning_result = warning_analysis(major_label_of_ablation, output_label_pc, case_num)
#     # certified from pc
#     robust_pc = certify_precomputed(prediction_map_pc, output_label_pc)
#     # certified from drs
#     output_label_drs, robust_drs = certified_drs(prediction_map_drs, ablation_size, patch_size)
#     # static analysis
#     static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
#     # label analysis
#     malicious_label_list_pc_not_include_output = pc_malicious_label_check(prediction_map_pc, output_label_pc)
#     malicious_label_dict_pc_with_location = pc_malicious_label_with_location(prediction_map_pc, output_label_pc,num_mask=args.num_mask)
#     malicious_label_list_drs_include_output = malicious_list_drs(prediction_map_drs, ablation_size, patch_size)
#     label_cert = malicious_list_compare(malicious_label_list_pc_not_include_output, malicious_label_list_drs_include_output,output_label_pc, output_label_drs)
#     stable_cert = static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
#     if not label_cert:
#         # malicious_label_list_drs_with_location = drs_malicious_label_with_location(prediction_map_drs, ablation_size,
#         #                                                                            patch_size)
#         location_cert=certified_with_location(malicious_label_dict_pc_with_location,suspect_column_list,patch_size,prediction_map_drs, ablation_size)
#         print(location_cert)
#         print(i)
#     else:
#         location_cert=True
#     if stable_cert == True and output_label_pc == label and warning_result == False:
#         stable_sample += 1
#     if static_cert_result == True and output_label_pc == label:
#         sta_certified_sample += 1
#     if robust_drs == True and output_label_pc == label:
#         cert_drs += 1
#     if label_cert == True and output_label_pc == label:
#         label_certified_sample += 1
#     if location_cert == True and output_label_pc == label:
#         location_certified_sample += 1
#     if output_label_pc == label:
#         if warning_result == True:
#             clean_sample_warning += 1
#         else:
#             clean_sample_nowarning += 1
#     if case_num==1 and output_label_pc == label:
#         OMA_detection_sample+=1
#     if robust_pc==True and output_label_pc == label:
#         pc_recovery_cert_sample+=1
# print("location_certified_sample " + str(location_certified_sample) + ' ' + str(location_certified_sample / NUM_IMG))
#
# print("sta_certified_sample " + str(sta_certified_sample) + ' ' + str(sta_certified_sample / NUM_IMG))
# print("label_certified_sample " + str(label_certified_sample) + ' ' + str(label_certified_sample / NUM_IMG))
# print("stable_sample " + str(stable_sample) + ' ' + str(stable_sample / NUM_IMG))
# print("OMA_detection_sample " + str(OMA_detection_sample) + ' ' + str(OMA_detection_sample / NUM_IMG))
# print("pc_recovery_cert_sample " + str(pc_recovery_cert_sample) + ' ' + str(pc_recovery_cert_sample / NUM_IMG))
#
# print("clean_sample_warning " + str(clean_sample_warning) + ' ' + str(clean_sample_warning / NUM_IMG))
# print("clean_sample_nowarning " + str(clean_sample_nowarning) + ' ' + str(clean_sample_nowarning / NUM_IMG))
# print("cert_drs " + str(cert_drs) + ' ' + str(cert_drs / NUM_IMG))

clean_sample_warning = 0
clean_sample_nowarning = 0
sta_certified_sample = 0
label_certified_sample = 0
not_sta_certified_sample = 0
not_label_certified_sample = 0
stable_sample = 0
cert_drs = 0
OMA_detection_sample = 0
pc_recovery_cert_sample = 0
location_certified_sample = 0
static_certified_detected_sample = 0
location_certified_detected_sample = 0
good_warning_in_not_cert_location = 0
correct_sample=0

clean_sample_warning_nowarning = 0
clean_sample_nowarning_nowarning = 0
sta_certified_sample_nowarning = 0
label_certified_sample_nowarning = 0
not_sta_certified_sample_nowarning = 0
not_label_certified_sample_nowarning = 0
stable_sample_nowarning = 0
cert_drs_nowarning = 0
OMA_detection_sample_nowarning = 0
pc_recovery_cert_sample_nowarning = 0
location_certified_sample_nowarning = 0
good_warning = 0
# if modify==True:
for i, (prediction_map_pc, label, prediction_map_drs) in enumerate(
        zip(prediction_map_list_pc, label_list, prediction_map_list_drs)):
    # print("label"+str(label))
    # init
    # generate a symmetric matrix from a triangle matrix
    prediction_map_pc = prediction_map_pc + prediction_map_pc.T - np.diag(np.diag(prediction_map_pc))

    # output label of PC
    # output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
    # output label of PC with modify
    output_label_pc, case_num = double_masking_precomputed_with_case_num_modify(prediction_map_pc)
    # calculate the majority
    major_label_of_ablation = majority_of_drs_single(prediction_map_drs)
    # warning analysis
    warning_result = warning_analysis_modify(major_label_of_ablation, output_label_pc, case_num)
    # certified from pc
    robust_pc = certify_precomputed(prediction_map_pc, output_label_pc)
    # certified from drs
    output_label_drs, robust_drs = certified_drs(prediction_map_drs, ablation_size, patch_size)
    # static analysis
    # static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    # label analysis
    malicious_label_list_pc_not_include_output = pc_malicious_label_check(prediction_map_pc, output_label_pc)
    malicious_label_dict_pc_with_location = pc_malicious_label_with_location(prediction_map_pc, output_label_pc,
                                                                             num_mask=args.num_mask)
    malicious_label_list_drs_include_output = malicious_list_drs(prediction_map_drs, ablation_size, patch_size)
    stable_cert = static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    if not robust_pc:
        # malicious_label_list_drs_with_location = drs_malicious_label_with_location(prediction_map_drs, ablation_size,
        #                                                                            patch_size)
        location_cert = certified_with_location(malicious_label_dict_pc_with_location, suspect_column_list, patch_size,
                                                prediction_map_drs, ablation_size)
        print(location_cert)
        print(i)
        label_cert = malicious_list_compare(malicious_label_list_pc_not_include_output,
                                            malicious_label_list_drs_include_output, output_label_pc, output_label_drs)
        static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    else:
        location_cert = False
        label_cert = False
        static_cert_result = False

    if stable_cert == True and output_label_pc == label and warning_result == False:
        stable_sample += 1
    if static_cert_result == True and output_label_pc == label:
        sta_certified_sample += 1
    if robust_drs == True and output_label_pc == label:
        cert_drs += 1
    if label_cert == True and output_label_pc == label:
        label_certified_sample += 1
    if location_cert == True and output_label_pc == label:
        location_certified_sample += 1
    if output_label_pc == label:
        if warning_result == True:
            clean_sample_warning += 1
        else:
            clean_sample_nowarning += 1
    if case_num == 1 and output_label_pc == label:
        OMA_detection_sample += 1
    if robust_pc == True and output_label_pc == label:
        pc_recovery_cert_sample += 1
    if (robust_pc == True or static_cert_result == True) and output_label_pc == label:
        static_certified_detected_sample += 1
    if (robust_pc == True or location_cert == True) and output_label_pc == label:
        location_certified_detected_sample += 1
    if (not (robust_pc == True or location_cert == True)) and (
            (not output_label_pc == label) and warning_result == True):
        good_warning_in_not_cert_location += 1

    if stable_cert == True and output_label_pc == label and warning_result == False:
        stable_sample_nowarning += 1
    if static_cert_result == True and output_label_pc == label and warning_result == False:
        sta_certified_sample_nowarning += 1
    if robust_drs == True and output_label_pc == label and warning_result == False:
        cert_drs_nowarning += 1
    if label_cert == True and output_label_pc == label and warning_result == False:
        label_certified_sample_nowarning += 1
    if location_cert == True and output_label_pc == label and warning_result == False:
        location_certified_sample_nowarning += 1
    if output_label_pc==label:
        correct_sample+=1
    # if output_label_pc == label:
    #     if warning_result == True:
    #         clean_sample_warning += 1
    #     else:
    #         clean_sample_nowarning += 1
    if case_num == 1 and output_label_pc == label and warning_result == False:
        OMA_detection_sample_nowarning += 1
    if robust_pc == True and output_label_pc == label and warning_result == False:
        pc_recovery_cert_sample_nowarning += 1
    if not output_label_pc == label and warning_result == False:
        good_warning += 1
print("modify")
print(
    "location_certified_sample " + str(location_certified_sample) + ' ' + str(location_certified_sample / NUM_IMG))
print("sta_certified_sample " + str(sta_certified_sample) + ' ' + str(sta_certified_sample / NUM_IMG))
print("label_certified_sample " + str(label_certified_sample) + ' ' + str(label_certified_sample / NUM_IMG))
print("stable_sample " + str(stable_sample) + ' ' + str(stable_sample / NUM_IMG))
print("OMA_detection_sample " + str(OMA_detection_sample) + ' ' + str(OMA_detection_sample / NUM_IMG))
print("pc_recovery_cert_sample " + str(pc_recovery_cert_sample) + ' ' + str(pc_recovery_cert_sample / NUM_IMG))
print("static_certified_detected_sample " + str(static_certified_detected_sample) + ' ' + str(
    static_certified_detected_sample / NUM_IMG))
print("location_certified_detected_sample " + str(location_certified_detected_sample) + ' ' + str(
    location_certified_detected_sample / NUM_IMG))
print("good_warning_in_not_cert_location " + str(good_warning_in_not_cert_location) + ' ' + str(
    good_warning_in_not_cert_location / NUM_IMG))

print("clean_sample_warning " + str(clean_sample_warning) + ' ' + str(clean_sample_warning / NUM_IMG))
print("clean_sample_nowarning " + str(clean_sample_nowarning) + ' ' + str(clean_sample_nowarning / NUM_IMG))
print("cert_drs " + str(cert_drs) + ' ' + str(cert_drs / NUM_IMG))
print("correct_sample " + str(correct_sample) + ' ' + str(correct_sample / NUM_IMG))

print("\n")
print("location_certified_sample_nowarning " + str(location_certified_sample_nowarning) + ' ' + str(
    location_certified_sample_nowarning / NUM_IMG))
print("sta_certified_sample_nowarning " + str(sta_certified_sample_nowarning) + ' ' + str(
    sta_certified_sample_nowarning / NUM_IMG))
print("label_certified_sample_nowarning " + str(label_certified_sample_nowarning) + ' ' + str(
    label_certified_sample_nowarning / NUM_IMG))
print("stable_sample_nowarning " + str(stable_sample_nowarning) + ' ' + str(stable_sample_nowarning / NUM_IMG))
print("OMA_detection_sample_nowarning " + str(OMA_detection_sample_nowarning) + ' ' + str(
    OMA_detection_sample_nowarning / NUM_IMG))
print("pc_recovery_cert_sample_nowarning " + str(pc_recovery_cert_sample_nowarning) + ' ' + str(
    pc_recovery_cert_sample_nowarning / NUM_IMG))

# print("clean_sample_warning_nowarning " + str(clean_sample_warning_nowarning) + ' ' + str(clean_sample_warning_nowarning / NUM_IMG))
# print("clean_sample_nowarning_nowarning " + str(clean_sample_nowarning_nowarning) + ' ' + str(clean_sample_nowarning_nowarning / NUM_IMG))
print("cert_drs_nowarning " + str(cert_drs_nowarning) + ' ' + str(cert_drs_nowarning / NUM_IMG))
print("good_warning " + str(good_warning) + ' ' + str(good_warning / NUM_IMG))

clean_sample_warning = 0
clean_sample_nowarning = 0
sta_certified_sample = 0
label_certified_sample = 0
not_sta_certified_sample = 0
not_label_certified_sample = 0
stable_sample = 0
cert_drs = 0
OMA_detection_sample = 0
pc_recovery_cert_sample = 0
location_certified_sample = 0
static_certified_detected_sample = 0
location_certified_detected_sample = 0
good_warning_in_not_cert_location = 0
correct_sample=0

clean_sample_warning_nowarning = 0
clean_sample_nowarning_nowarning = 0
sta_certified_sample_nowarning = 0
label_certified_sample_nowarning = 0
not_sta_certified_sample_nowarning = 0
not_label_certified_sample_nowarning = 0
stable_sample_nowarning = 0
cert_drs_nowarning = 0
OMA_detection_sample_nowarning = 0
pc_recovery_cert_sample_nowarning = 0
location_certified_sample_nowarning = 0
good_warning = 0

# else:
for i, (prediction_map_pc, label, prediction_map_drs) in enumerate(
        zip(prediction_map_list_pc, label_list, prediction_map_list_drs)):
    # init
    # generate a symmetric matrix from a triangle matrix
    prediction_map_pc = prediction_map_pc + prediction_map_pc.T - np.diag(np.diag(prediction_map_pc))

    # output label of PC
    # output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
    # output label of PC with modify
    output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
    # calculate the majority
    major_label_of_ablation = majority_of_drs_single(prediction_map_drs)
    # warning analysis
    warning_result = warning_analysis(major_label_of_ablation, output_label_pc, case_num)
    # certified from pc
    robust_pc = certify_precomputed(prediction_map_pc, output_label_pc)
    # certified from drs
    output_label_drs, robust_drs = certified_drs(prediction_map_drs, ablation_size, patch_size)
    # static analysis
    # static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    # label analysis
    malicious_label_list_pc_not_include_output = pc_malicious_label_check(prediction_map_pc, output_label_pc)
    malicious_label_dict_pc_with_location = pc_malicious_label_with_location(prediction_map_pc, output_label_pc,
                                                                             num_mask=args.num_mask)
    malicious_label_list_drs_include_output = malicious_list_drs(prediction_map_drs, ablation_size, patch_size)
    stable_cert = static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
    if not robust_pc:
        # malicious_label_list_drs_with_location = drs_malicious_label_with_location(prediction_map_drs, ablation_size,
        #                                                                            patch_size)
        # location_cert = certified_with_location(malicious_label_dict_pc_with_location, suspect_column_list,
        #                                         patch_size,
        #                                         prediction_map_drs, ablation_size)
        # print(location_cert)
        # print(i)
        # label_cert = malicious_list_compare(malicious_label_list_pc_not_include_output,
        #                                     malicious_label_list_drs_include_output, output_label_pc,
        #                                     output_label_drs)
        static_cert_result = static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs)
        location_cert = False
        label_cert = False
    else:
        location_cert = False
        label_cert = False
        static_cert_result = False

    if stable_cert == True and output_label_pc == label and warning_result == False:
        stable_sample += 1
    if static_cert_result == True and output_label_pc == label:
        sta_certified_sample += 1
    if robust_drs == True and output_label_pc == label:
        cert_drs += 1
    if label_cert == True and output_label_pc == label:
        label_certified_sample += 1
    if location_cert == True and output_label_pc == label:
        location_certified_sample += 1
    if output_label_pc == label:
        if warning_result == True:
            clean_sample_warning += 1
        else:
            clean_sample_nowarning += 1
    if case_num == 1 and output_label_pc == label:
        OMA_detection_sample += 1
    if robust_pc == True and output_label_pc == label:
        pc_recovery_cert_sample += 1
    if (robust_pc == True or static_cert_result == True) and output_label_pc == label:
        static_certified_detected_sample += 1
    if (robust_pc == True or location_cert == True) and output_label_pc == label:
        location_certified_detected_sample += 1
    if (not (robust_pc == True or location_cert == True)) and (
            (not output_label_pc == label) and warning_result == True):
        good_warning_in_not_cert_location += 1

    if stable_cert == True and output_label_pc == label and warning_result == False:
        stable_sample_nowarning += 1
    if static_cert_result == True and output_label_pc == label and warning_result == False:
        sta_certified_sample_nowarning += 1
    if robust_drs == True and output_label_pc == label and warning_result == False:
        cert_drs_nowarning += 1
    if label_cert == True and output_label_pc == label and warning_result == False:
        label_certified_sample_nowarning += 1
    if location_cert == True and output_label_pc == label and warning_result == False:
        location_certified_sample_nowarning += 1
    if output_label_pc == label:
        correct_sample += 1
    # if output_label_pc == label:
    #     if warning_result == True:
    #         clean_sample_warning += 1
    #     else:
    #         clean_sample_nowarning += 1
    if case_num == 1 and output_label_pc == label and warning_result == False:
        OMA_detection_sample_nowarning += 1
    if robust_pc == True and output_label_pc == label and warning_result == False:
        pc_recovery_cert_sample_nowarning += 1
    if not output_label_pc == label and warning_result == False:
        good_warning += 1

print("location_certified_sample " + str(location_certified_sample) + ' ' + str(location_certified_sample / NUM_IMG))
print("sta_certified_sample " + str(sta_certified_sample) + ' ' + str(sta_certified_sample / NUM_IMG))
print("label_certified_sample " + str(label_certified_sample) + ' ' + str(label_certified_sample / NUM_IMG))
print("stable_sample " + str(stable_sample) + ' ' + str(stable_sample / NUM_IMG))
print("OMA_detection_sample " + str(OMA_detection_sample) + ' ' + str(OMA_detection_sample / NUM_IMG))
print("pc_recovery_cert_sample " + str(pc_recovery_cert_sample) + ' ' + str(pc_recovery_cert_sample / NUM_IMG))
print("static_certified_detected_sample " + str(static_certified_detected_sample) + ' ' + str(
    static_certified_detected_sample / NUM_IMG))
print("location_certified_detected_sample " + str(location_certified_detected_sample) + ' ' + str(
    location_certified_detected_sample / NUM_IMG))
print("good_warning_in_not_cert_location " + str(good_warning_in_not_cert_location) + ' ' + str(
    good_warning_in_not_cert_location / NUM_IMG))

print("clean_sample_warning " + str(clean_sample_warning) + ' ' + str(clean_sample_warning / NUM_IMG))
print("clean_sample_nowarning " + str(clean_sample_nowarning) + ' ' + str(clean_sample_nowarning / NUM_IMG))
print("cert_drs " + str(cert_drs) + ' ' + str(cert_drs / NUM_IMG))
print("correct_sample " + str(correct_sample) + ' ' + str(correct_sample / NUM_IMG))


print("\n")
print("location_certified_sample_nowarning " + str(location_certified_sample_nowarning) + ' ' + str(
    location_certified_sample_nowarning / NUM_IMG))
print("sta_certified_sample_nowarning " + str(sta_certified_sample_nowarning) + ' ' + str(
    sta_certified_sample_nowarning / NUM_IMG))
print("label_certified_sample_nowarning " + str(label_certified_sample_nowarning) + ' ' + str(
    label_certified_sample_nowarning / NUM_IMG))
print("stable_sample_nowarning " + str(stable_sample_nowarning) + ' ' + str(stable_sample_nowarning / NUM_IMG))
print("OMA_detection_sample_nowarning " + str(OMA_detection_sample_nowarning) + ' ' + str(
    OMA_detection_sample_nowarning / NUM_IMG))
print("pc_recovery_cert_sample_nowarning " + str(pc_recovery_cert_sample_nowarning) + ' ' + str(
    pc_recovery_cert_sample_nowarning / NUM_IMG))

# print("clean_sample_warning_nowarning " + str(clean_sample_warning_nowarning) + ' ' + str(clean_sample_warning_nowarning / NUM_IMG))
# print("clean_sample_nowarning_nowarning " + str(clean_sample_nowarning_nowarning) + ' ' + str(clean_sample_nowarning_nowarning / NUM_IMG))
print("cert_drs_nowarning " + str(cert_drs_nowarning) + ' ' + str(cert_drs_nowarning / NUM_IMG))
print("good_warning " + str(good_warning) + ' ' + str(good_warning / NUM_IMG))
