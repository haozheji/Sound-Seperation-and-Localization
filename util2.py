# util2.py
import matplotlib.pyplot as plt

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.utils import read_image
import numpy as np

import cv2
import os
from os.path import join
import time
import random
import sys

people_index = voc_bbox_label_names.index('person')
iou_thres = 0.5
img_bs = 15

# replace a string with left or right
def rep_leri(inStr, flag):
    '''
    input the image dir and flag('l' or 'r')
    output a new dir
    '''
    ix = inStr.rfind('/')
    inStrTmp = inStr[:ix + 1] + flag + '_' + inStr[ix + 1:]
    #print "result is:", inStrTmp
    return inStrTmp

def cal_IOU(bbox1, bbox2):
    '''
    calculate the IOU from two bbox
    '''
    y1min, x1min, y1max, x1max = bbox1.astype(int)
    y2min, x2min, y2max, x2max = bbox2.astype(int)

    #the coord of the intersection rect
    xInt_min = max(x1min, x2min)
    xInt_max = min(x1max, x2max)
    yInt_min = max(y1min, y2min)
    yInt_max = min(y1max, y2max)

    #compute the area of the intersection rect
    interArea = max(0, xInt_max - xInt_min + 1) * max(0, yInt_max - yInt_min + 1)

    #compute the area of the two bboxes
    boxArea1 = max(0, x1max - x1min + 1) * max(0, y1max - y1min + 1)
    boxArea2 = max(0, x2max - x2min + 1) * max(0, y2max - y2min + 1)

    #compute the IOU
    iou = interArea /float(boxArea1 + boxArea2 - interArea)

    return iou

def crop_img(img,bbox):
    '''
    input bbox numpy bbox from the result
    '''
    ymin, xmin, ymax, xmax = bbox.astype(int)
    return img[ymin:ymax, xmin:xmax]

def crop_comple(img, bbox):
    '''
    compute the complementation bbox
    '''
    H, W, _ = img.shape
    ymin, xmin, ymax, xmax = bbox.astype(int)
    img1 = img[:, :xmin+1]
    img2 = img[:, xmax-1:]
    return [img1, img2]

def clean_data(bboxes, labels, scores):
    '''
    clean the data containing only people data
    '''
    new_bboxes = []
    new_labels = []
    new_scores = []
    for img_ix in range(len(bboxes)):
        # numpy array
        bbox = bboxes[img_ix]
        label = labels[img_ix]
        score = scores[img_ix]
        select_ix = [i for i, x in enumerate(list(label)) if x == people_index] # selected index for people
        new_bbox = []
        new_label = []
        new_score = []
        for ix in select_ix:
            new_bbox.append(bbox[ix])
            new_label.append(label[ix])
            new_score.append(score[ix])
        new_bboxes.append(np.clip(np.array(new_bbox),0,None))
        new_labels.append(np.array(new_label))
        new_scores.append(np.array(new_score))

    return new_bboxes, new_labels, new_scores

def seg_person(img_cv, model, img_dir):
    '''
    seg_person(img_list, model, img_ori, img_dir, bbox[img_ix])
    input two images
    return the cropped person and left/right label
    the image contains only one person
    if detected zeros person: return the imge
    if detected 1 person : return
    if detected more than 1: return the highest score
    '''
    img_ch = []
    for img in img_cv:
        cv2.imwrite('tmp_img.jpg', img)
        img_ch.append(read_image('tmp_img.jpg'))
    # model = SSD300(pretrained_model='voc0712').to_gpu(device='cuda:0')
    bboxes, labels, scores = clean_data(*model.predict(img_ch))
    l0,l1 = labels  # the label for two images
    # index 1 has a person in the right
    if l0.size == 0 and l1.size == 1:
        img_ix = 1
    elif l0.size == 1 and l1.size == 0:
        # index 0 has a person in the left
        img_ix = 0
    elif l0.size == 1 and l1.size == 1:
        s0, s1 = scores
        img_ix = int(s0<s1)
    else:
        return str(img_cv[0].shape[1] < img_cv[1].shape[1])     # True for right side

    imgCrop_new = crop_img(img_cv[img_ix], bboxes[img_ix][0])
    if img_ix == 0: # the other person is on the left
        cv2.imwrite(rep_leri(img_dir,'l'), imgCrop_new)
        return 'left'
    else:
        cv2.imwrite(rep_leri(img_dir,'r'), imgCrop_new)
        return 'right'


    # for img_ix in range(len(bboxes)):
    #     img = img_cv[img_ix]
    #     img_dir = image_dir_list[img_ix]
    #     box = bboxes[img_ix]
    #     label = labels[img_ix]
    #     score = scores[img_ix]
    #     # select_ix is the index inside box, label and scores
    #     select_ix = [i for i, x in enumerate(list(label)) if x == 14]   # select_ix index for people bboxes
    #     if len(select_ix) == 0:
    #         cv2.imwrite(img_dir.replace('.jpg','_one.jpg'), img)
    #     elif len(selected_ix) == 1:
    #         img_crop = crop_img(img, box[selected_ix[0]])
    #         cv2.imwrite(img_dir.replace('.jpg','_one.jpg'), img_crop)
    #     else:
    #         score_max_ix = score[selected_ix]
    #         img_crop =crop_img(img, box[selected_ix[]])#need to find the highest score ix]])
    #         cv2.imwrite(img_dir.replace('.jpg','one.jpg'), img_crop)

def seg_people(image_dir_list):
    '''
    segment two people from a list of images
    the rules are:
        1. if detected None: evenly split the image
        2. if detected one : check the rest and find the person
        3. if detected two : check the IOU if >0.5 select the higher score and do 2
        4. if more than two : compute the lowest IOU pair as the selected two people
    '''
    # start_time = time.time()

    # print image_dir_list
    # exit(1)
    count_num = [0,0,0,0]
    img_cv = []
    img_ch = []
    for img in image_dir_list:
        img_cv.append(cv2.imread(img))
        img_ch.append(read_image(img))
    # middle_time = time.time()-start_time
    # print "middle_time",middle_time
    model = SSD300(pretrained_model='voc0712')#.to_gpu(device = 'cuda')
    bboxes, labels, scores = clean_data(*model.predict(img_ch)) # containing only people data

    for img_ix in range(len(bboxes)):   # img_ix is the index from the list of images
        img = img_cv[img_ix]
        img_dir = image_dir_list[img_ix]
        print "process: ", img_dir
        bbox = bboxes[img_ix]
        label = labels[img_ix]
        score = scores[img_ix]
        # select_ix = [i for i, x in enumerate(list(label)) if x == 14]   # select_ix index for selected bboxes

        # if None detected people
        fEmpty = label.size # the flag for empty
        if fEmpty == 0:
            # midVal = int(img.shape[1]/2)
            # rimgCrop = img[:, midVal:]
            # limgCrop = img[:, :midVal]
            # cv2.imwrite(rep_leri(img_dir,'r'), rimgCrop)
            # cv2.imwrite(rep_leri(img_dir,'l'), limgCrop)
            count_num[0] += 1

        elif fEmpty == 2: # detected exactly two people
            rimg_ix = int(bbox[0][1] < bbox[1][1])  # rimg_ix=1 if <
            limg_ix = int(not(rimg_ix))
            # rimg_ix = select_ix[int(box[select_ix[0]][1] < box[select_ix[1]][1])] # compare the xmin with both images
            # limg_ix = select_ix[int(not(rimg_ix))]
            if cal_IOU(bbox[rimg_ix],bbox[limg_ix]) < iou_thres:
                # print "I am here!", bbox[limg_ix]
                rimg_crop = crop_img(img, bbox[rimg_ix])
                limg_crop = crop_img(img, bbox[limg_ix])
                cv2.imwrite(rep_leri(img_dir,'r'), rimg_crop)
                cv2.imwrite(rep_leri(img_dir,'l'), limg_crop)
                count_num[2] += 1

            else:
                # only one is needed the other is the rest
                if (score[rimg_ix] > score[limg_ix]):
                    img_ix = rimg_ix
                else:
                    img_ix = limg_ix
                # given the left and right complement image
                img_crop_ori = crop_img(img, bbox[img_ix])
                img_list = crop_comple(img, bbox[img_ix])
                crop_pos = seg_person(img_list, model)
                if crop_pos =='left':   # the new image is on the left
                    cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
                elif crop_pos =='right':
                    cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
                else:
                    if crop_pos == 'True':
                        cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
                        cv2.imwrite(rep_leri(img_dir,'r'), img_list[1])
                    else:
                        cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
                        cv2.imwrite(rep_leri(img_dir,'l'), img_list[0])

                count_num[1] += 1

        elif fEmpty == 1:
            img_crop_ori = crop_img(img, bbox[0])
            img_list = crop_comple(img, bbox[0])
            crop_pos = seg_person(img_list, model, img_dir)
            if crop_pos =='left':   # the new image is on the left
                cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
            elif crop_pos =='right':
                cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
            else:
                if crop_pos == 'True':
                    cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
                    cv2.imwrite(rep_leri(img_dir,'r'), img_list[1])
                else:
                    cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
                    cv2.imwrite(rep_leri(img_dir,'l'), img_list[0])
            count_num[1]+=1

        else: # more than two
            # # check every two of them and return the lowest Iou
            # ix_list = range(len(bbox))
            # comb_tup = [(x,y) for x in ix_list for y in ix_list if x!=y]
            # for entry in comb_tup:
            #     if (entry[1], entry[0]) in comb_tup:
            #         comb_tup.remove((entry[1], entry[0]))
            # # the comb_tup is every combination index
            # min_IOU = 100
            # min_ix_tup = (-1,-1)
            # for entry in comb_tup:
            #     tmp_iou = cal_IOU(bbox[entry[0]], bbox[entry[1]])
            #     if(tmp_iou < min_IOU):
            #         min_IOU = tmp_iou
            #         min_ix_tup = entry
            # rimg_ix = int(bbox[min_ix_tup[0]][1] < bbox[min_ix_tup[1]][1])  # rimg_ix=1 if <
            # limg_ix = int(not(rimg_ix))
            # rimg_ix = min_ix_tup[rimg_ix]
            # limg_ix = min_ix_tup[limg_ix]
            # # rimg_ix = select_ix[int(box[select_ix[0]][1] < box[select_ix[1]][1])] # compare the xmin with both images
            # # limg_ix = select_ix[int(not(rimg_ix))]
            # if cal_IOU(bbox[rimg_ix],bbox[limg_ix]) < iou_thres:
            #     # print "I am here!", bbox[limg_ix]
            #     rimg_crop = crop_img(img, bbox[rimg_ix])
            #     limg_crop = crop_img(img, bbox[limg_ix])
            #     cv2.imwrite(rep_leri(img_dir,'r'), rimg_crop)
            #     cv2.imwrite(rep_leri(img_dir,'l'), limg_crop)
            #     count_num[3] += 1

            # else:
            #     # only one is needed the other is the rest
            #     if (score[rimg_ix] > score[limg_ix]):
            #         img_ix = rimg_ix
            #     else:
            #         img_ix = limg_ix
            #     # given the left and right complement image
            #     img_crop_ori = crop_img(img, bbox[img_ix])
            #     img_list = crop_comple(img, bbox[img_ix])
            #     crop_pos = seg_person(img_list, model)
            #     if crop_pos =='left':   # the new image is on the left
            #         cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
            #     elif crop_pos =='right':
            #         cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
            #     else:
            #         if crop_pos == 'True':
            #             cv2.imwrite(rep_leri(img_dir,'l'), img_crop_ori)
            #             cv2.imwrite(rep_leri(img_dir,'r'), img_list[1])
            #         else:
            #             cv2.imwrite(rep_leri(img_dir,'r'), img_crop_ori)
            #             cv2.imwrite(rep_leri(img_dir,'l'), img_list[0])

            #     count_num[1] += 1
            # # print " More than two in this images!", img_dir
            # # exit()
            print "More than two(Deprecated image, TA don't worry!)", img_dir
    # print count_num
    print "Finished a folder"
    # print labels
    # all_time = time.time()-start_time
    # print "all_time", all_time

'''
bboxes, labels scores examples

[array([[  56.817432,   24.472895, 1089.8677  ,  810.6728  ],
       [  26.574247, 1126.0125  , 1096.2502  , 2560.4207  ]],
      dtype=float32), array([[ 142.04045,  677.9337 ,  718.9993 , 1067.8767 ],
       [ 130.45692,  407.03348,  701.7813 ,  719.6197 ]], dtype=float32)]
[array([14, 14], dtype=int32), array([14, 14], dtype=int32)]
[array([0.9851527, 0.9726449], dtype=float32), array([0.96293247, 0.8406958 ], dtype=float32)]

'''

def img_batch(imageDir):
    '''
    input the image dir input e.g. test7Path
    return a list containing a list with img_bs=10 images per list from the same env
    '''
    # print rep_leri('/home/wcf/doh.jpg','l')
    # exit(1)
    outImgList = []
    folderList = [join(imageDir, x) for x in os.listdir(imageDir)]
    for folder in folderList:
        imgBatch = []   # the length is img_bs
        imgAll = sorted(os.listdir(folder))
        imgAllLen = len(imgAll)
        # print i[-1]
        imgAll = sorted(random.sample(imgAll[int(imgAllLen/3) : int(2 * imgAllLen / 3)], img_bs))
        imgBatch = [join(folder, x) for x in imgAll]
        outImgList.append(imgBatch)
    return outImgList


if __name__ == '__main__':
    # Read an RGB image and return it in CHW format.
   #  rootPath = '/home/sdd/std_dataset/homework'
   # # test7Path = join(rootPath, 'testset7', 'testimage')
   #  test25Path = join(rootPath, 'testset25', 'testimage')
   # # test7list = img_batch(test7Path)
   #  test25list = img_batch(test25Path)
   #  #for entry in test7list:
   #   #   seg_people(entry)
   #  for entry in test25list:
   #      seg_people(entry)
    arg1 = sys.argv[1]
    if  os.path.isdir(arg1):
        data_path = join(arg1,'testimage')
    else:
        data_path = join(os.getcwd(),arg1, 'testimage')
    testlist = img_batch(data_path)
    for entry in testlist:
        seg_people(entry)

