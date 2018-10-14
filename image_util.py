import argparse
import sys
import glob
import os
from PIL import Image

def __list_files(src):
    pattern = src + os.path.sep + '*.jpg'
    print(pattern)
    list = glob.glob(pattern)
    return list

def __center_crop(files, out, size):
    for f in files:
        image = Image.open(f)
        width, height = image.size
        if width >= height:
            left = (width - height)/2
            image = image.crop((left,0,width-left,height)) #left,upper,right,lower
        else:
            top = (height-width)/2
            image = image.crop((0,top,width,height-top))
        image = image.resize((size,size))
        filename = out + os.path.sep + os.path.basename(f)        
        image.save(filename)

def __pad(files, out, size):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="dirctory of src images")
    parser.add_argument("-o", '--out', help='destination dirctory : default out')
    parser.add_argument("-m", '--mode', help='transform mode : center crop  or pad to square : default center', choices=['crop','pad'])
    parser.add_argument('-s', '--size' ,help='size of the image (px) : default 96 ', type=int)

    args = parser.parse_args()

    print(args)

    if args.src is None:
        args.src = 'img'+os.path+'*.jpg'

    if args.out is None:
        args.out = 'out'
    
    if args.mode is None:
        args.mode = 'crop'
    
    if args.size is None:
        args.size = 96

    files = __list_files(args.src)

    os.makedirs(args.out, exist_ok=True)
    if args.mode == 'crop':
        __center_crop(files,args.out,args.size)
    else:
        __pad(files,args.out,args.size)

    print('{} files transformed.'.format(len(files)))