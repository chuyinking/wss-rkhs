import os
import shutil

path = '../../../Datasets/ACDC/tz/small/All Images/saliency/train all'
with open('splits/plain/sal900.txt', 'w') as f:
    for file in os.listdir(path):
        fn = file[:-4]
        f.write(fn+'.jpg '+fn+'.png '+fn+'.pt\n')
f.close()

# check_path = '../../../Datasets/VOC2012/SegmentationScribble'
# save_path = '../../../Datasets/ACDC/tz/small'
# for file in os.listdir(path):
#     fn = file[:-4]
#     scribble_file = os.path.join(check_path, fn+'.png')
#     if not os.path.exists(scribble_file):
#         print(fn)
#     else:
#         save_file = os.path.join(save_path, fn+'.png')
#         shutil.copyfile(scribble_file, save_file)

# id_path = './splits/plain/sal20.txt'
# image_path = '../../../Datasets/ACDC/tz/small'
# save_path = '../records/generated_images/rkhs_NN/500 saliency/eval'
# with open(id_path, 'r') as f:
#     ids = f.read().splitlines()
#     for id in ids:
#         img_file = id.split(' ')[0]
#         prob_file = id.split(' ')[1][:-4]+'_prob.png'
#         thre_file = id.split(' ')[1][:-4]+'_thre.png'
#         shutil.copyfile(os.path.join(image_path, 'saliency/val', img_file), os.path.join(save_path, img_file))
#         shutil.copyfile(os.path.join(image_path, 'L2RK/more', prob_file), os.path.join(save_path, prob_file))
#         shutil.copyfile(os.path.join(image_path, 'L2RK/more', thre_file), os.path.join(save_path, thre_file))

#im_path = '../../../Datasets/VOC2012/SegmentationClassAug/2008_000181.png'