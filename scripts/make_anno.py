import imageio
import numpy as np
import os, cv2
import paddlehub as hub
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

def to3channels(mask):
    h, w = mask.shape[:2]
    mask3 = np.zeros((h,w,3), dtype=mask.dtype)
    mask3[:,:,0] = mask
    mask3[:,:,1] = mask
    mask3[:,:,2] = mask
    return mask3

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

def process_gif(image_path, pose_estimation):
    action = image_path.split('/')[-1].split('.gif')[0]
    # read query gif 
    frames, bboxes, size = read_gif(image_path)
    N = len(frames)
    for i in range(N):
        image = frames[i]
        src_bbox = bboxes[i]
        angle = angles['%s_%d'%(action, i)] if '%s_%d'%(action, i) in angles else 0

        image = rotate(image, angle)
        image = pad(image, 1.2)

        cv2.imwrite('anno/%s_%d.png'%(action, i), image)
        result = pose_estimation.predict(image)
        poses = parse_openpose_result(result)

        np.save('anno/%s_%d.npy'%(action, i), poses)

pose_estimation = hub.Module(name='openpose_body_estimation')
root = './ryu'
for fname in os.listdir(root):
    if(fname.endswith('.gif')):
        print(fname)
        process_gif(os.path.join(root, fname), pose_estimation)
