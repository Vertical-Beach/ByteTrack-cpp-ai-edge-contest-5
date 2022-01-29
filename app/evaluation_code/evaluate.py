# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:19:44 2021

@author: n.aoi
"""

import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser


class Correspondence():
    def __init__(self, threshold):
        self.cumu_matches = {}
        self.temp_misses = {}
        self.threshold = threshold
    

    def count_fp_fn_idsw(self, true, pred): 
        results = {}
        matches = {}

        intersection_categories = set(true).intersection(set(pred))
        for intersection_category in intersection_categories:
            gt = true[intersection_category]
            pr = pred[intersection_category]
            matches[intersection_category] = {}
            results[intersection_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            results[intersection_category]['GT'] = len(gt)
            if intersection_category in self.cumu_matches:
                cumu_matches = self.cumu_matches[intersection_category]
                                                
                m = self.find_temp_match(gt, pr, cumu_matches, self.threshold)
                matches[intersection_category].update(m)

                gt_unmatched = list(filter(lambda x: x['id'] not in matches[intersection_category] and x['id'] in cumu_matches, gt))
                pr_unmatched = list(filter(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                m = self.find_match(gt_unmatched, pr_unmatched, self.threshold)
                results[intersection_category]['IDSW'] += len(m)
                matches[intersection_category].update(m)

                new_objects = list(filter(lambda x: x['id'] not in cumu_matches, gt))
                pr_objects = list(filter(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                m = self.find_match(new_objects, pr_objects, self.threshold)
                matches[intersection_category].update(m)
                    
                results[intersection_category]['FN'] += sum(map(lambda x: x['id'] not in matches[intersection_category], gt))
                results[intersection_category]['FP'] += sum(map(lambda x: x['id'] not in matches[intersection_category].values(), pr))
            else:                                
                m = self.find_match(gt, pr, self.threshold)
                matches[intersection_category].update(m)
                    
                results[intersection_category]['FN'] += sum(map(lambda x: x['id'] not in matches[intersection_category], gt))
                results[intersection_category]['FP'] += sum(map(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                
        
        pred_true_difference_categories = set(pred).difference(set(true))
        for pred_true_difference_category in pred_true_difference_categories:
            results[pred_true_difference_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            pr = pred[pred_true_difference_category]
            results[pred_true_difference_category]['FP'] += len(pr)
            
        true_pred_difference_categories = set(true).difference(set(pred))
        for true_pred_difference_category in true_pred_difference_categories:
            results[true_pred_difference_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            gt = true[true_pred_difference_category]
            results[true_pred_difference_category]['GT'] += len(gt)
            results[true_pred_difference_category]['FN'] += len(gt)
        
        
        ## update cumulative matches
        for c, cumu_match in self.cumu_matches.items():
            if c in matches:
                self.cumu_matches[c].update(matches[c])
        new_categories = set(matches).difference(set(self.cumu_matches))
        for new_category in new_categories:
            self.cumu_matches[new_category] = matches[new_category]
        
        ## update misses(fn, fp, idsw)
        self.temp_misses = results
    

    def find_match(self, gt_objects, pr_objects, threshold):
        result = {}
        if len(gt_objects) and len(pr_objects):
            mat = []
            for gt_object in gt_objects:
                mat.append([compute_iou_bb(pr_object['box2d'], gt_object['box2d']) for pr_object in pr_objects])
            profit_array = np.array(mat)
            cost_array = 1 - np.array(mat)
            
            row_ind, col_ind = linear_sum_assignment(cost_array)
            matches = np.array((row_ind, col_ind)).T
            result.update({gt_objects[i]['id']: pr_objects[j]['id'] for i, j in matches if profit_array[i][j] >= threshold})
        
        return result


    def find_temp_match(self, gt_objects, pr_objects, matches, threshold):
        result = {}
        for g_id, p_id in matches.items():
            g_object = list(filter(lambda x: x['id'] == g_id, gt_objects))
            p_object = list(filter(lambda x: x['id'] == p_id, pr_objects))
            if len(g_object) == 1 and len(p_object) == 1:
                iou = compute_iou_bb(p_object[0]['box2d'], g_object[0]['box2d'])
                if iou >= threshold:
                    result[g_id] = p_id

        return result


def compute_iou_bb(pred_bb, true_bb):
    pred_area = (pred_bb[2] - pred_bb[0])*(pred_bb[3] - pred_bb[1])
    true_area = (true_bb[2] - true_bb[0])*(true_bb[3] - true_bb[1])
    intersection_x = max(min(pred_bb[2], true_bb[2]) - max(pred_bb[0], true_bb[0]), 0)
    intersection_y = max(min(pred_bb[3], true_bb[3]) - max(pred_bb[1], true_bb[1]), 0)
    intersection_area = intersection_x*intersection_y
    union_area = pred_area + true_area - intersection_area

    if union_area > 0:
        return intersection_area/union_area
    else:
        return 0


def mota(traj_true, traj_pred, threshold):
    corr = Correspondence(threshold = threshold)
    scores = {}
    for gt, pr in zip(traj_true, traj_pred):
        corr.count_fp_fn_idsw(gt, pr)
        for c, r in corr.temp_misses.items():
            if c not in scores:
                scores[c] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            scores[c]['FP'] += corr.temp_misses[c]['FP']
            scores[c]['FN'] += corr.temp_misses[c]['FN']
            scores[c]['IDSW'] += corr.temp_misses[c]['IDSW']
            scores[c]['GT'] += corr.temp_misses[c]['GT']
    mota = 0
    gt_non_zero = 0
    for c,r in scores.items():
        if r['GT'] > 0:
            ss = 1 - (r['FP'] + r['FN'] + r['IDSW'])/r['GT']
            print(' ', c, ss)
            mota += ss
            gt_non_zero += 1
        
    if gt_non_zero != 0:
        return mota/gt_non_zero
    else:
        return -10000


def MOTA(ans, sub, threshold):
    ans_files = set(ans).intersection(set(sub))
    s = 0
    for ans_file in sorted(ans_files):
        print(ans_file)
        traj_true = ans[ans_file]
        traj_pred = sub[ans_file]
        s += mota(traj_true, traj_pred, threshold)
        print('\n')

    return s/len(ans_files)


def validate_gt(ans, eval_categories, min_area, min_count):
    new_ans = {}
    for ans_file, seq in ans.items():
        # category check
        category_filtered = []
        for element in seq:
            new_element = {}
            for c, ann in element.items():
                if c in eval_categories:
                    new_element[c] = ann
                else:
                    print('Removing {} in {}'.format(c, ans_file))
            category_filtered.append(new_element)

        # bbox area check
        filtered_annotation = []
        for element in category_filtered:
            new_element = {}
            for c, ann in element.items():
                object_list = []
                for bbox in ann:
                    area = abs(bbox['box2d'][2]-bbox['box2d'][0])*abs(bbox['box2d'][3]-bbox['box2d'][1])
                    if area >= min_area:
                        object_list.append(bbox)
                    else:
                        print('{}: Removing {}(id {}) whose area is less than {}'.format(ans_file, c, bbox['id'], min_area))
                if len(object_list):
                    new_element[c] = object_list
            filtered_annotation.append(new_element)

        # object count check
        frame_counts = {}
        for element in filtered_annotation:
            for c, bboxes in element.items():
                for bbox in bboxes:
                    i = bbox['id']
                    b = bbox['box2d']
                    if c not in frame_counts:
                        frame_counts[c]={}
                    if i not in frame_counts[c]:
                        frame_counts[c][i]=0
                    frame_counts[c][i]+=1
        new_annotation = []
        for element in filtered_annotation:
            filtered_element = {}
            for c, bboxes in element.items():
                if c not in filtered_element:
                    filtered_element[c] = []
                for bbox in bboxes:
                    if frame_counts[c][bbox['id']]>=min_count:
                        filtered_element[c].append(bbox)
                    else:
                        print('{}: Removing {}(id {}) whose count is less than {}'.format(ans_file, c, bbox['id'], min_count))
                if len(filtered_element[c])==0:
                    del filtered_element[c]
            new_annotation.append(filtered_element)

        # check object in the sequence
        flag = 0
        for element in new_annotation:
            if len(element):
                flag = 1
                break
        if flag:
            new_ans[ans_file] = new_annotation
        else:
            print('Removing {} because no ground truth exists.'.format(ans_file))

    if len(new_ans):
        return new_ans
    else:
        return None


def validate_pr(ans, sub, eval_categories):
    # category check
    for sub_file, seq in sub.items():
        categories = set()
        for element in seq:
            categories.update(element)
        diff = categories.difference(eval_categories)
        if len(diff):
            print('Invalid categories found in {}: {}'.format(sub_file, diff))
            return None

    # sequence length check
    ans_files = set(ans).intersection(set(sub))
    for ans_file in ans_files:
        gt = ans[ans_file]
        pr = sub[ans_file]
        if len(gt)!=len(pr):
            print('The length of sequence does not match in {}'.format(ans_file))
            return None

    return sub


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prediction-file', default = 'predictions.json')
    parser.add_argument('--answer-file', default = 'ans.json')
    parser.add_argument('--threshold', default = 0.5)

    return parser.parse_args()


def main():
    args = parse_args()

    # read required files
    try:
        with open(args.answer_file) as f:
            ans = json.load(f)
        with open(args.prediction_file) as f:
            sub = json.load(f)
    except Exception as e:
        print(e)
        return None

    # validation
    print('-------- Validation --------')
    eval_categories = {'Car', 'Pedestrian'}
    min_area = 1024
    min_count = 3
    ans = validate_gt(ans, eval_categories, min_area, min_count)
    sub = validate_pr(ans, sub, eval_categories)

    # compute MOTA
    threshold = args.threshold
    if ans is not None and sub is not None:
        print('Validation OK\n')
        print('-------- Evaluation --------')
        score = MOTA(ans, sub, threshold)
        print('Overall Score: {}'.format(score))
    else:
        print('Validation NG\n')
        print('Please fix ans or sub.')


if __name__ == '__main__':
    main()