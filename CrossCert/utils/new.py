import numpy as np
import torch

torch.set_printoptions(profile='full')


def majority_of_mask_single(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]

    return majority_pred


def majority_of_drs_single(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred, cnt = np.unique(prediction_map, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]

    return majority_pred


def disagree_second_mask(prediction_map_old):
    # second-round masking
    pred_mutant_list, cnt_mutant = np.unique(prediction_map_old, return_counts=True)
    # # get majority prediction and disagreer prediction
    # sorted_idx = np.argsort(cnt_mutant)
    # majority_pred = pred_mutant_list[sorted_idx][-1]

    return pred_mutant_list, cnt_mutant


def certified_warning_detection(prediction_map_old, prediction_map_masks, prediction_label, t):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask_mutant = np.diag(prediction_map_old)
    pred_one_mask_mutant_list, cnt_pred_one_mask_mutant = np.unique(pred_one_mask_mutant, return_counts=True)
    pred_one_mask = np.diag(prediction_map_masks)
    pred_one_mask_list, cnt_pred_one_mask = np.unique(pred_one_mask, return_counts=True)

    if len(pred_one_mask_mutant_list) == 1:  # unanimous agreement in the first-round masking
        pred_two_mask_mutant_list, cnt_pred_two_mask_mutant = disagree_second_mask(prediction_map_old)

        for idx_pred_two_mask_mutant in range(len(pred_two_mask_mutant_list)):
            for idx_pred_one_mask in range(len(pred_one_mask_list)):
                if pred_two_mask_mutant_list[idx_pred_two_mask_mutant] == pred_one_mask_list[
                    idx_pred_one_mask] and not prediction_label == pred_two_mask_mutant_list[idx_pred_two_mask_mutant]:
                    if cnt_pred_one_mask[idx_pred_one_mask] > t:
                        return "cannot_certified_warning"  # fail
        return "certified_warning"  # success
    else:
        # calculate label of masks
        # first-round
        for pred_one_mask_mutant in pred_one_mask_mutant_list:
            for idx_pred_one_mask in range(len(pred_one_mask_list)):
                if pred_one_mask_mutant == pred_one_mask_list[
                    idx_pred_one_mask] and not prediction_label == pred_one_mask_mutant:
                    if cnt_pred_one_mask[idx_pred_one_mask] > t:
                        return "cannot_certified_warning"  # fail
        # second-round
        pred_two_mask_mutant_list, cnt_pred_two_mask_mutant = disagree_second_mask(prediction_map_old)

        for idx_pred_two_mask_mutant in range(len(pred_two_mask_mutant_list)):
            for idx_pred_one_mask in range(len(pred_one_mask_list)):
                if pred_two_mask_mutant_list[idx_pred_two_mask_mutant] == pred_one_mask_list[
                    idx_pred_one_mask] and not prediction_label == pred_two_mask_mutant_list[idx_pred_two_mask_mutant]:
                    if cnt_pred_one_mask[idx_pred_one_mask] > t:
                        return "cannot_certified_warning"  # fail
        return "certified_warning"  # success


def certified_nowarning_detection(prediction_map_masks, prediction_label, t, delta=16):
    pred_one_mask = np.diag(prediction_map_masks)
    pred_list, cnt = np.unique(pred_one_mask, return_counts=True)
    for idx in range(len(pred_list)):
        if pred_list[idx] == prediction_label:
            if cnt[idx] > t + 2 * delta:
                return "certified_no_warning"
    return "cannot_certified_no_warning"


def warning_detection(prediction_map_masks, prediction_label, t, delta=16):
    pred_one_mask = np.diag(prediction_map_masks)
    pred_list, cnt = np.unique(pred_one_mask, return_counts=True)
    for idx in range(len(pred_list)):
        if pred_list[idx] == prediction_label:
            if cnt[idx] <= t + delta:
                return "warning"
            else:
                return "no_warning"
    return "warning"


def certified_warning_drs(malicious_label_list, prediction_map_drs, prediction_label, t):
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    for pred_idx in range(len(pred_list)):
        for malicious_label in malicious_label_list:
            if pred_list[pred_idx] == malicious_label and not pred_list[pred_idx] == prediction_label:
                if cnt[pred_idx] > t:
                    return "cannot_certified_warning_drs"
    return "certified_warning_drs"


def warning_drs(prediction_map_drs, prediction_label, t, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    for pred_idx in range(len(pred_list)):
        if pred_list[pred_idx] == prediction_label:
            if cnt[pred_idx] <= t + delta:
                return "warning"
    return "no_warning"


def pc_malicious_label(prediction_map_old, prediction_label):
    malicious_label_list = []
    pred_one_mask_mutant = np.diag(prediction_map_old)
    pred_one_mask_mutant_list, cnt_pred_one_mask_mutant = np.unique(pred_one_mask_mutant, return_counts=True)
    pred_two_mask_mutant_list, cnt_pred_two_mask_mutant = np.unique(prediction_map_old, return_counts=True)

    if len(pred_one_mask_mutant_list) == 1:  # agreement in the first-round masking
        for pred_two_mask_mutant in pred_two_mask_mutant_list:
            if not pred_two_mask_mutant == prediction_label:
                malicious_label_list.append(pred_two_mask_mutant)
        return malicious_label_list  # success
    else:
        for pred_one_mask_mutant in pred_one_mask_mutant_list:
            if not pred_one_mask_mutant == prediction_label:
                malicious_label_list.append(pred_one_mask_mutant)
        for pred_two_mask_mutant in pred_two_mask_mutant_list:
            if not pred_two_mask_mutant == prediction_label:
                malicious_label_list.append(pred_two_mask_mutant)
        return malicious_label_list  # success


def pc_malicious_label_check(prediction_map_old, prediction_label):
    malicious_label_list = []
    for idx_one in range(len(prediction_map_old)):
        for idx_two in range(len(prediction_map_old)):
            label_for_mutant = prediction_map_old[idx_one][idx_two]
            if not label_for_mutant == prediction_label:
                malicious_label_list.append(label_for_mutant)
    return malicious_label_list


def pc_malicious_label_with_location(prediction_map_old, prediction_label, num_mask=6):
    malicious_label_dict = {}
    for idx_one in range(len(prediction_map_old)):
        for idx_two in range(len(prediction_map_old)):
            label_for_mutant = prediction_map_old[idx_one][idx_two]
            if not label_for_mutant == prediction_label:
                key_name = idx_one * num_mask * num_mask + idx_two
                malicious_label_dict[key_name] = label_for_mutant
    return malicious_label_dict


def certified_drs(prediction_map_drs, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    # get majority prediction and disagreer prediction
    if len(sorted_value) > 1:
        gap = sorted_value[-1] - sorted_value[-2]
    else:
        gap = sorted_value[-1]
    if gap > 2 * delta:
        return majority_pred, True
    else:
        return majority_pred, False
    #
    # malicious_list = []
    # delta = ablation_size + patch_size - 1
    # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    # sorted_idx = np.argsort(cnt)
    # majority_pred = pred_list[sorted_idx][-1]
    # sorted_value = np.sort(cnt)
    # max_vote_value = sorted_value[-1]
    # for idx in range(len(cnt)):
    #     if abs(cnt[idx] - max_vote_value) <= delta:
    #         malicious_list.append(pred_list[idx])
    # return malicious_list


# def drs_malicious_label_with_location(prediction_map_drs, ablation_size, patch_size, num_classses=1000):
#     malicious_label_dict = []
#     # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
#     # sorted_idx = np.argsort(cnt)
#     # majority_pred_now = pred_list[sorted_idx][-1]
#     delta = ablation_size + patch_size - 1
#     for idx_one in range(len(prediction_map_drs)):
#         prediction_map_drs_copy=prediction_map_drs.copy()
#         malicious_label_list=[]
#         for label in range(num_classses):
#             prediction_map_drs_copy[idx_one:idx_one+delta]=label
#             pred_list, cnt = np.unique(prediction_map_drs_copy, return_counts=True)
#             sorted_idx = np.argsort(cnt)
#             majority_pred = pred_list[sorted_idx][-1]
#             if majority_pred not in malicious_label_list:
#                 malicious_label_list.append(majority_pred)
#         malicious_label_dict.append(malicious_label_list)
#     return malicious_label_dict

# def drs_malicious_label_with_location_fast(idx_one,prediction_map_drs, ablation_size, patch_size, label):
#     malicious_label_dict = []
#     # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
#     # sorted_idx = np.argsort(cnt)
#     # majority_pred_now = pred_list[sorted_idx][-1]
#     delta = ablation_size + patch_size - 1
#     prediction_map_drs_copy=prediction_map_drs.copy()
#     if idx_one<0:
#         prediction_map_drs_copy[len(prediction_map_drs_copy)+idx_one:len(prediction_map_drs_copy)]=label
#         prediction_map_drs_copy[0:idx_one+delta]=label
#         if idx_one+delta<0:
#             print("wrong")
#     elif idx_one+delta>len(prediction_map_drs_copy):
#         prediction_map_drs_copy[idx_one:idx_one + delta] = label #question
#         prediction_map_drs_copy[0:idx_one + delta-len(prediction_map_drs_copy)] = label
#     else:
#         prediction_map_drs_copy[idx_one:idx_one+delta]=label
#     pred_list, cnt = np.unique(prediction_map_drs_copy, return_counts=True)
#     sorted_idx = np.argsort(cnt)
#     majority_pred = pred_list[sorted_idx][-1]
#     return majority_pred

def drs_malicious_label_with_location_fast_jiaozhun(idx_one, prediction_map_drs, ablation_size, patch_size, label):
    malicious_label_dict = []
    # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    # sorted_idx = np.argsort(cnt)
    # majority_pred_now = pred_list[sorted_idx][-1]
    idx_one = idx_one - ablation_size + 1
    delta = ablation_size + patch_size - 1
    prediction_map_drs_copy = prediction_map_drs.copy()
    if idx_one < 0:
        prediction_map_drs_copy[len(prediction_map_drs_copy) + idx_one:len(prediction_map_drs_copy)] = label
        prediction_map_drs_copy[0:idx_one + delta] = label
        if idx_one + delta < 0:
            print("wrong")
    elif idx_one + delta > len(prediction_map_drs_copy):
        # prediction_map_drs_copy[idx_one:idx_one + delta] = label #question
        prediction_map_drs_copy[idx_one:len(prediction_map_drs_copy)] = label
        prediction_map_drs_copy[0:idx_one + delta - len(prediction_map_drs_copy)] = label
    else:
        prediction_map_drs_copy[idx_one:idx_one + delta] = label
    pred_list, cnt = np.unique(prediction_map_drs_copy, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    return majority_pred


def mask_ablation_for_all(num_mask, mask_list):
    maskfree_list = []
    for idx_a in range(num_mask * num_mask):
        for idx_b in range(num_mask * num_mask):
            maskfree_list.append(mask_ablation_for_single(idx_a, idx_b, mask_list))
    # print("maskfree_list")
    # print(maskfree_list)
    return maskfree_list


def mask_ablation_for_single(idx_a, idx_b, mask_list):
    maskfree = torch.ones_like(mask_list[0])
    # print("mask_list")
    # print(mask_list)
    for mask_idx in range(len(mask_list)):
        if mask_idx == idx_a or mask_idx == idx_b:
            continue
        maskfree = torch.where(mask_list[mask_idx], maskfree, torch.tensor(0.).cuda())
    return maskfree


def suspect_column_list_cal(maskfree_all_list):
    maskfree_list_malicious_column_list = []
    for maskfree in maskfree_all_list:
        maskfree_malicious_column_list = []
        for idx in range(maskfree.shape[2]):
            column = maskfree[:, :, idx, :]
            if column.any():
                maskfree_malicious_column_list.append(idx)
        maskfree_list_malicious_column_list.append(maskfree_malicious_column_list)
    # print("maskfree_list_malicious_column_list")
    # print(maskfree_list_malicious_column_list)
    return maskfree_list_malicious_column_list


def suspect_column_list_cal_fix(mask_list):
    maskfree_list_malicious_column_list = []
    for mask_1 in mask_list:
        for mask_2 in mask_list:
            mask_malicious_column_list = []
            for idx in range(mask_1.shape[2]):
                column_1 = ~mask_1[:, :, idx, :]
                column_2 = ~mask_2[:, :, idx, :]
                if column_1.any() or column_2.any():
                    mask_malicious_column_list.append(idx)
            maskfree_list_malicious_column_list.append(mask_malicious_column_list)
    return maskfree_list_malicious_column_list


def check_maskfree_empty(maskfree_all_list):
    count = 0
    idx = 0
    for maskfree in maskfree_all_list:
        if not maskfree.any():
            count += 1
            print(idx)
        idx += 1
    return count


def certified_with_location(malicious_label_dict_pc_with_location, suspect_column_list_pc, patch_size,
                            prediction_map_drs, ablation_size):
    for idx in malicious_label_dict_pc_with_location:
        suspect_column_l_pc = suspect_column_list_pc[idx]
        malicious_label = malicious_label_dict_pc_with_location.get(idx)
        for suspect_column_pc in suspect_column_l_pc:
            if suspect_column_pc + patch_size - 1 in suspect_column_l_pc:
                output_label = drs_malicious_label_with_location_fast_jiaozhun(suspect_column_pc, prediction_map_drs,
                                                                               ablation_size, patch_size,
                                                                               malicious_label)
                if output_label == malicious_label:
                    return False
            else:
                continue

    return True

    # good version
    # for idx in malicious_label_dict_pc_with_location:
    #     suspect_column_l_pc=suspect_column_list_pc[idx]
    #     malicious_label=malicious_label_dict_pc_with_location.get(idx)
    #     for suspect_column_pc in suspect_column_l_pc:
    #         for idx_drs in range(suspect_column_pc-patch_size+1,suspect_column_pc+1):
    #             output_label=drs_malicious_label_with_location_fast(idx_drs, prediction_map_drs, ablation_size, patch_size, malicious_label)
    #             if output_label==malicious_label:
    #                 return False
    # return True

    # for idx in malicious_label_dict_pc_with_location:
    #     suspect_column_l_pc=suspect_column_list_pc[idx]
    #     malicious_label=malicious_label_dict_pc_with_location.get(idx)
    #     for suspect_column_pc in suspect_column_l_pc:
    #         for idx_drs in range(suspect_column_pc-patch_size+1,suspect_column_pc+1):
    #             drs_malicious_list=malicious_label_list_drs_with_location[idx_drs]
    #             if not len(drs_malicious_list)==0 and malicious_label in drs_malicious_list:
    #                 return False
    # return True


def malicious_list_drs(prediction_map_drs, ablation_size, patch_size, num_classses=1000):
    malicious_list = []
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    max_vote_value = sorted_value[-1]
    for idx in range(len(cnt)):
        if abs(cnt[idx] - max_vote_value) <= 2 * delta:
            malicious_list.append(pred_list[idx])
    if max_vote_value <= 2 * delta:
        for label in range(num_classses):
            malicious_list.append(label)
    return malicious_list


def malicious_list_compare(malicious_label_list_pc_not_include_output, malicious_label_list_drs_include_output,
                           output_label_pc, output_label_drs):
    # if not output_label_pc==output_label_drs:
    #     return False
    for label_drs in malicious_label_list_drs_include_output:
        for label_pc in malicious_label_list_pc_not_include_output:
            if label_drs == label_pc:
                return False
    return True


def double_masking_precomputed_with_case_num(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)

    if len(pred) == 1:  # unanimous agreement in the first-round masking
        return pred[0], 1  # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask, dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp, pred_one_mask == dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == dis
        if np.all(tmp):
            return dis, 2  # Case II: disagreer prediction

    return majority_pred, 3  # Case III: majority prediction

def double_masking_precomputed_with_case_num_modify(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred, cnt = np.unique(pred_one_mask, return_counts=True)

    if len(pred) == 1:  # unanimous agreement in the first-round masking
        return pred[0], 1  # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask, dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp, pred_one_mask == dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in range(len(pred_one_mask)):
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i] == dis
        if np.all(tmp):
            return dis, 2  # Case II: disagreer prediction

    return majority_pred, 3  # Case III: majority prediction

def warning_analysis_modify(drs_major, pc_major, pc_case):
    warn = False
    if not drs_major == pc_major:
        warn = True
        return warn
    if pc_case == 3:
        warn = True
        return warn
    return warn

def warning_analysis(drs_major, pc_major, pc_case):
    warn = False
    if not drs_major == pc_major:
        warn = True
        return warn
    return warn
