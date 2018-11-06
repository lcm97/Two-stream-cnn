"""
    This module is called initial because it initializes all request
    from this node. It will simulates a (224, 224, 3) size image data
    packet and send to the first node in the distributed system and wait
    for the response from the last layer.
"""
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from multiprocessing import Queue
from threading import Thread
from Data import sliding_window

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

# data packet format definition
PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Initializer:
    """
        Singleton factory for initializer. The Initializer module has two timers.
        The node_timer is for recording statistics for block1 layer model inference
        time. The timer is for recording the total inference time from last
        fully connected layer.

        Attributes:
            queue: Queue for storing available block1 models devices.
            start: Start time of getting a frame.
            count: Total Number of frames gets back.
            node_total: Total layer-wise time.
            node_count: Total layer-wise frame count.
    """
    instance = None

    def __init__(self):
        self.ip = dict()
        self.start = 0.0
        self.count = 0
        self.node_total = 0
        self.node_count = 1

    def timer(self):
        # count == 0 then means the node just starts, so start the timer.
        if self.count == 0:
            self.start = time.time()
        else:
            print 'total time: {:.3f} sec'.format((time.time() - self.start) / self.count)
        self.count += 1

    def node_timer(self, mode, interval):
        """
            Print out time used by a specific module.

            Args:
                mode: A string for node mode.
                interval: A float for time lapse.
        """
        self.node_total += interval
        print '{:s}: {:.3f}'.format(mode, self.node_total / self.node_count)
        self.node_count += 1

    @classmethod
    def create_init(cls):
        """ Utilize singleton design pattern to create single instance. """
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance


def send_request(bytestr1, bytestr2, mode1, mode2, tag=''):
    """
        This function sends data to next layer. It will pop an available
        next layer device IP address defined at IP table, and send data
        to that IP. After, it will put the available IP back.

        Args:
            bytestr1: The encoded byte string for frames.
            bytestr2: The encoded byte string for optical frames.
            mode1: Specify next layer option.
            mode2: Specify next layer option.
    """
    init = Initializer.create_init()

    queue1 = init.ip[mode1]
    queue2 = init.ip[mode2]
    addr1 = queue1.get()
    addr2 = queue2.get()
    """"""
    client1 = ipc.HTTPTransceiver(addr1, 12345)
    requestor1 = ipc.Requestor(PROTOCOL, client1)

    data = dict()
    data['input'] = bytestr1
    data['next'] = mode1
    data['tag'] = tag

    start = time.time()
    requestor1.request('forward', data)
    print 'sending data to spatial'
    end = time.time()

    init.node_timer(mode1, end - start)

    client1.close()
    queue1.put(addr1)
    """"""
    client2 = ipc.HTTPTransceiver(addr2, 12345)
    requestor2 = ipc.Requestor(PROTOCOL, client2)

    data['input'] = bytestr2
    data['next'] = mode2
    data['tag'] = tag

    start = time.time()
    requestor2.request('forward', data)
    print 'sending data to temporal'
    end = time.time()

    init.node_timer(mode2, end - start)

    client2.close()
    queue2.put(addr2)   


def master():
    """
        Master function for real time model inference. A basic while loop
        gets one frame at each time. It appends a frame to deque every time
        and pop the least recent one if the length > maximum.
    """
    init = Initializer.create_init()
    while True:
        data = sliding_window()
        frames = data[0]
        opt_flows = data[1]

        frames = frames.astype(dtype=np.uint8)
        opt_flows = opt_flows.astype(dtype=np.uint8)
        Thread(target=send_request, args=(frames.tobytes(), opt_flows.tobytes(), 'spatial', 'temporal', 'initial')).start()
        time.sleep(0.6)


class Responder(ipc.Responder):
    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
            This functino is invoked by do_POST to handle the request. Invoke handles
            the request and get response for the request. This is the key of each node.
            All models forwarding and output redirect are done here. Because the invoke
            method of initializer only needs to receive the data packet, it does not do
            anything in the function and return None.

            Args:
                msg: Meta data.
                req: Contains data packet.

            Returns:
                None

            Raises:
                AvroException: if the data does not have correct syntac defined in Schema
        """
        if msg.name == 'forward':
            init = Initializer.create_init()
            try:
                print 'initial node gets data'
                init.timer()
                return
            except Exception, e:
                print 'Error', e.message
        else:
            raise schema.AvroException('unexpected message:', msg.getname())


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
            Handle request from other devices.
            do_POST is automatically called by ThreadedHTTPServer. It creates a new
            responder for each request. The responder generates response and write
            response to data sent back.
        """
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ Handle requests in separate thread. """


def main():
    init = Initializer.create_init()
    # read ip resources from config file
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        address = address['node']
        init.ip['spatial'] = Queue()
        init.ip['temporal'] = Queue()
        for addr in address['spatial']:
            if addr == '#':
                break
            init.ip['spatial'].put(addr)
        for addr in address['temporal']:
            if addr == '#':
                break
            init.ip['temporal'].put(addr)
    # listen on port 9999 for model inference result
    server = ThreadedHTTPServer(('0.0.0.0', 9999), Handler)
    server.allow_reuse_address = True
    Thread(target=server.serve_forever, args=()).start()

    master()


if __name__ == '__main__':
    main()
