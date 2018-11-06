import numpy as np
from keras.utils import to_categorical


def sliding_window():
    """sliding window stimulation, the window size is 16 which
    generates 16 long frame and optical flows"""
    static_frames = []
    opt_flow_stacks = []

    for i in range(16):
        static_frame = np.random.rand(12, 16, 3)
        static_frames.append(static_frame)

        opt_flow_stack = np.random.rand(12, 16, 20)
        opt_flow_stacks.append(opt_flow_stack)

    return np.array(static_frames), np.array(opt_flow_stacks)


def get_class_one_hot():
    y = to_categorical(np.random.randint(256, size=(16, 1)), num_classes=256)
    return y


