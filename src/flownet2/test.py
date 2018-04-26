import time	
import argparse
import os
from ..net import Mode
from .flownet2 import FlowNet2

FLAGS = None


def main():
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)

    # Train on the data
    net.test(
        checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
        input_a_path=FLAGS.input_a,
        input_b_path=FLAGS.input_b,
        out_path=FLAGS.out,
        batchTestDirectory= FLAGS.batchTestDirectory,
    )

def testInit(checkpoint):
    print('start initialization')
    net = FlowNet2(mode=Mode.TEST)
    net.testInit(checkpoint)
    return net


def fn2Opflow(net, input_a, input_b):
    #net = FlowNet2(mode=Mode.TEST)
    return net.fn2Opflow(input_a, input_b) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--input_b',
        type=str,
        required=True,
        help='Path to second image'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output flow result'
    )
    parser.add_argument(
        '--batchTestDirectory',
        type=str,
        help='Directory of input images with the form *-0.png and *-1.png'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.input_b):
        raise ValueError('image_b path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    #if not os.path.isdir(FLAGS.batchTestDirectory):
    #    raise ValueError('batch directory must exist')
    main()
