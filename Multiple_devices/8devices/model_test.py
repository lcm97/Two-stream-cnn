import Model as ml
import numpy as np
from Data import sliding_window
from collections import deque
import time

"""**********************Initial**********************"""
test_data = sliding_window()
"""(16,12,16,3)"""
frames = test_data[0]
"""(16,12,16,20)"""
opt_flows = test_data[1]

frames = frames.astype(dtype=np.uint8)
opt_flows = opt_flows.astype(dtype=np.uint8)

"""convert to bytes for transporting"""
frames = frames.tobytes()
opt_flows = opt_flows.tobytes()

"""*************Spatial CNNs and Temporal CNNs******************"""
frames = np.fromstring(frames, np.uint8).reshape(16, 12, 16, 3)
opt_flows = np.fromstring(opt_flows, np.uint8).reshape(16, 12, 16, 20)

start = time.time()
s_model = ml.spatial_model_multi()
s_output = s_model.predict(np.array([frames]))
end = time.time()
print 'spatial cnn costs:'
print (end-start)
t_model = ml.temporal_model_multi()
t_output = t_model.predict(np.array([opt_flows]))
"""(16, 256)x2"""

s_output = s_output.tostring()
t_output = t_output.tostring()

"""*******************Maxpoolings and fc_1(8k)***************"""
s_output = np.fromstring(s_output, np.float32).reshape(16, 256)
t_output = np.fromstring(t_output, np.float32).reshape(16, 256)

_input = deque()
_input.append(s_output)
_input.append(t_output)

start = time.time()
mp_model = ml.maxpoolings()
s_output = mp_model.predict(np.array([_input[0]]))
t_output = mp_model.predict(np.array([_input[1]]))
"""(15, 256)x2"""

con_model = ml.temporal_pyramid_concate()
X = con_model.predict([np.array([s_output]), np.array([t_output])])
"""(2, 15, 256)"""
"""******maxpoolings done!******"""

inter_dense = ml.fc_1()
X1 = inter_dense.predict(X)
X2 = inter_dense.predict(X)
end = time.time()

print "block1 costs:"
print end - start
print X.shape
X1 = X1.tostring()
X2 = X2.tostring()

"""********************fc_2***********************"""

X1 = np.fromstring(X1, np.float32).reshape(1, 4096)
X2 = np.fromstring(X2, np.float32).reshape(1, 4096)
start = time.time()

X = np.concatenate([X1, X2], 1)
inter_dense = ml.fc_2()
X1 = inter_dense.predict(X)
X2 = inter_dense.predict(X)
"""4k x 2"""
end = time.time()
print "block2 costs:"
print end - start
X1 = X1.tostring()
X2 = X2.tostring()
"""*********************fc_3********************"""
X1 = np.fromstring(X1, np.float32).reshape(1, 4096)
X2 = np.fromstring(X2, np.float32).reshape(1, 4096)
start = time.time()

X = np.concatenate([X1, X2], 1)
inter_dense = ml.fc_3()
X = inter_dense.predict(X)
"""51"""
end = time.time()
print "block3 costs:"
print end - start
print X

