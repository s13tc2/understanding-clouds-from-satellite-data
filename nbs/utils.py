import numpy as np

def rle2mask(mask_rle, shape=(2100,1400), shrink=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T[::shrink,::shrink]

def rle2bb(rle):
    if rle=='': return (0,0,0,0)
    mask = rle2mask(rle)
    z = np.argwhere(mask==1)
    mn_x = np.min( z[:,0] )
    mx_x = np.max( z[:,0] )
    mn_y = np.min( z[:,1] )
    mx_y = np.max( z[:,1] )
    return (mn_x,mn_y,mx_x-mn_x,mx_y-mn_y)

def bb2dice(rle,bb,shape=(2100,1400)):
    mask1 = rle2mask(rle,shape)
    mask2 = np.zeros((shape[1],shape[0]))
    mask2[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]=1
    union = np.sum(mask1) + np.sum(mask2)
    if union==0: return 1
    intersection = np.sum(mask1*mask2)
    return 2.*intersection/union

def bb2rle(b,shape=(525,350)):
    bb = [0,0,0,0]
    bb[0] = np.min(( np.max((int(b[0]),0)),shape[1] ))
    bb[1] = np.min(( np.max((int(b[1]),0)),shape[0] ))
    bb[2] = np.min(( np.max((int(b[2]),0)),shape[1]-bb[0] ))
    bb[3] = np.min(( np.max((int(b[3]),0)),shape[0]-bb[1] ))
    z = np.ones((bb[3]*2))*bb[2]
    z[::2] = np.arange(bb[3]) * shape[1] + (shape[1]*bb[1]+bb[0]+1)
    z = z.astype(int)
    return ' '.join(str(x) for x in z)

def dice_coef6(y_true_rle, y_pred_prob, y_pred_rle,th):
    if y_pred_prob<th:
        if y_true_rle=='': return 1
        else: return 0
    else:
        y_true_f = rle2mask(y_true_rle,shrink=4)
        y_pred_f = rle2mask(y_pred_rle,shape=(525,350))
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        if union==0: return 1
        intersection = np.sum(y_true_f * y_pred_f)
        return 2. * intersection / union
    
def mask2contour(mask, width=5):
    w = mask.shape[1]
    h = mask.shape[0]
    
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    
    return np.logical_or(mask2,mask3) 