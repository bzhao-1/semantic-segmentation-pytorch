from scipy.io import loadmat
import scipy.io as sio


# create new mat file 
new_colors = [
  [128, 64, 128],
  [244, 35, 232],
  [70, 70, 70],
  [102, 102, 156],
  [190, 153, 153],
  [153, 153, 153],
  [250, 170, 30],
  [220, 220, 0],
  [107, 142, 35],
  [152, 251, 152],
  [70, 130, 180],
  [220, 20, 60],
  [255, 0, 0],
  [0, 0, 142],
  [0, 0, 70],
  [0, 60, 100],
  [0, 80, 100],
  [0, 0, 230],
  [119, 11, 32],
  [110, 190, 160],
  [170, 120, 50],
  [55, 90, 80],
  [45, 60, 150],
  [157, 234, 50],
  [81, 0, 81],
  [150, 100, 100],
  [230, 150, 140],
  [180, 165, 180],
  [0, 0, 0],
]


sio.savemat('data/color_29.mat', {'colors': new_colors})

