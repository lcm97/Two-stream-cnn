from Data import get_static_frame_and_stacked_opt_flows
from Two_stream_model import two_stream_model
import numpy as np
from keras import backend as K
import time


def main():
    test_data = get_static_frame_and_stacked_opt_flows()
    model = two_stream_model()
    start = time.time()
    for i in range(50):
        result = model.predict([np.expand_dims(test_data[0], axis=0),
                                np.expand_dims(test_data[1], axis=0)])
    end = time.time()
    total_time = end - start
    K.clear_session()


if __name__ == '__main__':
    main()
