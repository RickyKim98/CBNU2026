import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Simple OpenCV drawing and crop tool')
    parser.add_argument('--path', default='lenna.png', help='Input image path')
    parser.add_argument('--output', default='output.png', help='Output image path for save command')
    return parser.parse_args()
