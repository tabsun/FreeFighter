import os, sys
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import paddlehub as hub

import imageio
import numpy as np
from skimage.measure import label
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

angles = {'RYU1_beAttacked_fall_1': 90,
          'RYU1_beAttacked_fall_2': 180,
          'RYU1_beAttacked_fall_3': 90,
          'RYU1_fall_down_0': 90,
          'RYU1_fall_down_1': 90,
          'RYU1_fall_down_2': 90,
          'RYU1_heavy_kick_0': 90,
          'RYU1_heavy_kick_1': 90,
          'RYU1_heavy_kick_2': 90,
          'RYU1_heavy_kick_3': 90,
          'RYU1_jump_back_3': 90,
          'RYU1_jump_back_4': 180,
          'RYU1_jump_back_5': 270,
          'RYU1_jump_forward_3': 270,
          'RYU1_jump_forward_4': 180,
          'RYU1_jump_forward_5': 90,
          'RYU1_somesault_up_0': 90,
          'RYU1_somesault_up_1': 180,
          'RYU1_somesault_up_2': 180,
          'RYU1_somesault_up_3': 270}

def resize_fix(image, size):
    h, w = image.shape[:2]
    dw, dh = size
    scale = min(float(dw)/w, float(dh)/h)
    return cv2.resize(image, (int(w*scale), int(h*scale)))

def rotate(image, angle):
    assert(angle in [0, 90, 270, 180])
    if(angle == 0):
        return image
    elif(angle == 90):
        for i in range(3):
            image = np.rot90(image)
        return image
    elif(angle == 180):
        return np.rot90(np.rot90(image))
    else:
        return np.rot90(image)

def pad(image, scale):
    h, w = image.shape[:2]
    std_size = int(max(w, h) * scale)
    full = np.zeros((std_size, std_size, 3), dtype=np.uint8)
    left, top = (std_size-w)//2, (std_size-h)//2
    full[top:top+h, left:left+w, :] = image
    return full

def to3channels(mask):
    h, w = mask.shape[:2]
    mask3 = np.zeros((h,w,3), dtype=mask.dtype)
    mask3[:,:,0] = mask
    mask3[:,:,1] = mask
    mask3[:,:,2] = mask
    return mask3

def enlarge_bbox(bbox, scale, size):
    h, w = size
    t, b, l, r = bbox
    width, height = r-l, b-t
    scale = (scale - 1.) / 2.
    t -= int(height * scale)
    b += int(height * scale)
    l -= int(width * scale)
    r += int(width * scale)
    
    t = max(0, min(h-1, t))
    b = max(0, min(h-1, b))
    l = max(0, min(w-1, l))
    r = max(0, min(w-1, r))
    return [t, b, l, r]

def left_largest_patch(mask):
    label_map, num = label(mask, neighbors=8, background=0, return_num=True)
    high_val = np.max(mask)

    largest_area = -float('inf')
    largest_label_id = -1
    for i in range(1, num+1):
        cur_area = np.sum(label_map==i)
        if(cur_area > largest_area):
            largest_area = cur_area
            largest_label_id = i
    mask[label_map!=largest_label_id] = 0
    mask[label_map==largest_label_id] = high_val
    return mask

def select_largest_pose(poses):
    pose = None
    max_area = -float('inf')
    for cur_pose in poses:
        temp_pose = cur_pose[cur_pose != -1].reshape(-1,2)
        left, top = np.min(temp_pose, axis=0)
        right, bottom = np.max(temp_pose, axis=0)
        area = (bottom-top)*(right-left)
        if(area > max_area):
            max_area = area
            pose = cur_pose
    return pose
            
def read_gif(path):
    reader = imageio.get_reader(path)
    ims = []
    try:
        for im in reader:
            ims.append(im)
    except RuntimeError:
        pass
    reader.close()
    assert(len(ims) == 1)
    image = ims[0]
    size = image.shape

    h, w = image.shape[:2]
    frames = []
    boxes = []
    mask = image[:,:,-1]
    label_map, num = label(mask, neighbors=8, background=0, return_num=True)
    for label_id in range(1, num+1):
        mask = (label_map == label_id)
        t, b, l, r = get_bbox(mask)

        person_image = np.zeros((h, w, 3), dtype=np.uint8)
        np.copyto(person_image, image[:,:,:-1], where=(to3channels(mask)>0))
        
        frames.append(person_image[t:b, l:r, :])
        boxes.append([t,b,l,r])

    _, frames, bboxes = zip(*sorted(zip([(box[2]+box[3])/2 for box in boxes], frames, boxes)))

    return frames, bboxes, size

def read_source_image(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
    mask = human_seg.segmentation(images=[image])[0]['data']
    image[:,:,0][mask == 0] = 0 
    image[:,:,1][mask == 0] = 0 
    image[:,:,2][mask == 0] = 0 

    left = np.argmax(np.max(mask, axis=0)!=0)
    right = w - np.argmax(np.max(mask, axis=0)[::-1]!=0)
    top = np.argmax(np.max(mask, axis=1)!=0)
    bottom = h - np.argmax(np.max(mask, axis=1)[::-1]!=0)
    height = bottom - top + 1
    width = right - left + 1
    std_size = max(width, height)
    cx, cy = (left+right)//2, (top+bottom)//2
    left, right = cx - std_size//2, cx + std_size//2
    top, bottom = cy - std_size//2, cy + std_size//2
    return image[top:bottom, left:right, ::-1] 

def normalize(pose):
    pose = pose[:, :2]
    mask = (pose != -1).reshape(-1,2)
    mask_un = (pose == -1).reshape(-1,2)
    temp_pose = pose[mask].reshape(-1,2)

    cx, cy = np.mean(temp_pose, axis=0)
    left, top = np.min(temp_pose, axis=0)
    right, bottom = np.max(temp_pose, axis=0)
    pose = pose.astype(np.float)
    dist = float(min(right-left, bottom-top))
    pose[:, 0] -= left
    pose[:, 0] /= float(right-left)
    pose[:, 1] -= top
    pose[:, 1] /= float(bottom-top)
    pose[mask_un] = -1
    return pose

def calc_dist(p, q):
    assert(p.shape == q.shape)
    mask = np.bitwise_and(p != -1, q != -1).reshape(p.shape)
    unalign_dist = p.shape[0] - np.sum(mask)/2
    p = p[mask].reshape(-1,2)
    q = q[mask].reshape(-1,2)
    dists = np.linalg.norm(p-q, ord=2, axis=1)
    return np.mean(dists) + unalign_dist

def find_best_pose(q, pool):
    idx = -1
    min_dist = float('inf')
    for i, p in enumerate(pool):
        dist = calc_dist(q, p)
        if(dist < min_dist):
            min_dist = dist
            idx = i
    return idx
    
def parse_openpose_result(result):
    subset = result['subset']
    pts = result['candidate']
    if(len(subset) == 0):
        return np.zeros((0, 18), dtype=np.float)

    poses = np.ones((subset.shape[0], 18, 2), dtype=np.int) * -1
    for i in range(subset.shape[0]):
        for index, pt_id in enumerate(subset[i,:18]):
            pt_id = int(pt_id)
            if(pt_id != -1):
                poses[i, index, :] = pts[pt_id, :2]
    return poses

def parse_mpii_result(result):
    #parts = ['left_ankle', 'left_knee', 'left_hip', 'right_hip', 'right_knee', 'right_ankle', 'pelvis', 'thorax', 'upper_neck', 'head_top', 'right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist']
    parts = ['left_ankle', 'left_knee', 'left_hip', 'right_hip', 'right_knee', 'right_ankle', 'upper_neck', 'head_top', 'right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist']
    pose = np.zeros((len(parts),2), dtype=np.float)
    for i, part in enumerate(parts):
        pt = result[0]['data'][part]
        pose[i, 0] = pt[0]
        pose[i, 1] = pt[1]
    return pose

def parse_result(result):
    candidate = result['candidate']
    subsets = result['subset']
    subset = subsets[0]
    
    pose = np.zeros((18,4), dtype=np.float)
    for i in range(18):
        if(subset[i] == -1):
            continue
        pose[i, :] = candidate[int(subset[i]), :]
    center = np.mean(pose, axis=0)
    for i in range(18):
        if(pose[i][0] == 0 and pose[i][1] == 0):
            pose[i] = center
    return pose

def get_bbox(mask):
    h, w = mask.shape
    mask[mask > 0] = 255
    cols = np.max(mask, axis=0)
    rows = np.max(mask, axis=1)
    left = np.argmax(cols)
    right = w - np.argmax(cols[::-1])
    top = np.argmax(rows)
    bottom = h - np.argmax(rows[::-1])
    return [top, bottom, left, right]

def generate_pose_gif(image_path, pose_estimation, human_seg, pose_pool, image_pool):
    action = image_path.split('/')[-1].split('.gif')[0]
    # read query gif 
    frames, bboxes, size = read_gif(image_path)
    gif = np.ones(size, dtype=np.uint8)*255
    gif[:,:,-1] = 0
    N = len(frames)
    for i in range(N):
        image = frames[i]
        src_bbox = bboxes[i]
        angle = angles['%s_%d'%(action, i)] if '%s_%d'%(action, i) in angles else 0

        image = rotate(image, angle)
        image = pad(image, 1.2)

        #result = pose_estimation.keypoint_detection(images=[image], use_gpu=True)
        #pose = parse_mpii_result(result)
        #result = pose_estimation.predict(image)
        #poses = parse_openpose_result(result)

        #assert(len(poses) == 1)
        #pose = poses[0]

        # read pose from annotations
        pose = np.load(os.path.join('anno', '%s_%d.npy'%(action, i)))
        #result = model.predict(image)
        #pose = parse_result(result)
        query_pose = normalize(pose)
        
        idx = find_best_pose(query_pose, pose_pool)
        target_image = image_pool[idx]
        mask = human_seg.segmentation(images=[target_image], use_gpu=True)[0]['data'] 
        mask[mask > 0] = 255
        mask = left_largest_patch(mask)

        t, b, l, r = get_bbox(mask)
        st, sb, sl, sr = src_bbox
        person_image = np.ones((target_image.shape[0], target_image.shape[1], 3), dtype=np.uint8) * 255
        np.copyto(person_image, target_image, where=(to3channels(mask)>0))
        rgb = resize_fix(rotate(person_image[t:b, l:r, :], (360-angle)%360), (sr-sl, sb-st))
        bg  = resize_fix(rotate(mask[t:b, l:r], (360-angle)%360), (sr-sl, sb-st))
        bg[bg!=0] = 255
        offset_x = sl + (sr-sl-rgb.shape[1])//2
        offset_y = st + (sb-st-rgb.shape[0])//2
        gif[:,:,:-1][offset_y:offset_y+rgb.shape[0],offset_x:offset_x+rgb.shape[1],:] = rgb
        gif[:,:,-1][offset_y:offset_y+rgb.shape[0],offset_x:offset_x+rgb.shape[1]] = bg
    return gif

def augment(image):
    images = []
    images.append(image)
    images.append(image[:,::-1,:])
    return images

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_dir", default='./StreetFighter/images/RYU1', help="path to source image")
    parser.add_argument("--search_video", default='./mp4/dance.mp4', help="path to driving video")
    parser.add_argument("--dest_dir", default='output', help='directory to save the output gif images')
 
    opt = parser.parse_args()
    print(opt)

    pose_estimation = hub.Module(name='openpose_body_estimation')
    #pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")

    # extract pose pool and image_pool from search_video    
    reader = imageio.get_reader(opt.search_video)
    driving_video = []
    step = 1
    max_side = 640
    index = 0
    try:
        for im in tqdm(reader):
            if(index % step == 0):
                image = im[..., ::-1]
                h, w = image.shape[:2]
                if(max(w, h) > 640):
                    scale = 640. / max(w, h) 
                    nh, nw = int(h*scale), int(w*scale)
                    image = cv2.resize(image, (nw, nh))
                driving_video.append(image)
            index += 1
    except RuntimeError:
        pass
    reader.close()

    pose_pool = []
    image_pool = []
    for image in tqdm(driving_video):
        for cur_image in augment(image):
            result = pose_estimation.predict(cur_image)
            poses = parse_openpose_result(result)
            if(len(poses) > 0):
                pose = select_largest_pose(poses)
                #result = pose_estimation.keypoint_detection(images=[cur_image], use_gpu=True)
                #pose = parse_mpii_result(result)
                pose_pool.append(normalize(pose))
                image_pool.append(cur_image)

    if(not os.path.exists(opt.dest_dir)):
        os.makedirs(opt.dest_dir)

    for fname in os.listdir(opt.source_dir):
        if(fname.endswith('.gif') and 'fire' not in fname):
            print(fname)
            gif = generate_pose_gif(os.path.join(opt.source_dir, fname), pose_estimation, human_seg, pose_pool, image_pool)
            dst_path = os.path.join(opt.dest_dir, fname)
            temp_path = dst_path.replace('.gif', '.png')
            cv2.imwrite(temp_path, gif)
            os.rename(temp_path, dst_path)
