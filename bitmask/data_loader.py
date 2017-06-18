#!/usr/bin/env python3

import numpy as np

def get_batch(batch_size = 100, image_size = 20, region_size = 3):
    input_images = np.random.random((batch_size,image_size,image_size))
    output_images = np.random.random((batch_size,image_size,image_size))
    input_mask = np.zeros((batch_size,image_size,image_size))
    output_mask = np.zeros((batch_size,image_size,image_size))

    input_mask_locations = np.random.randint(1+image_size-region_size,size = (batch_size,2))
    output_mask_locations = np.random.randint(1+image_size-region_size,size = (batch_size,2))

    for i in range(batch_size):
        in_x = input_mask_locations[i,0]
        in_y = input_mask_locations[i,1]
        input_mask[i, in_x : in_x + region_size, in_y : in_y + region_size] = np.ones((region_size,region_size))
        out_x = output_mask_locations[i,0]
        out_y = output_mask_locations[i,1]
        output_mask[i, out_x : out_x + region_size, out_y : out_y + region_size] = np.ones((region_size,region_size))
        output_images[i, out_x : out_x + region_size, out_y : out_y + region_size] = \
        input_images[i, in_x : in_x + region_size, in_y : in_y + region_size]

    return [input_images, output_images, input_mask, output_mask]

def main():
    [a, b, c, d]  = get_batch(batch_size = 1, image_size = 5)
    print(a)
    print(b)
    print(c)
    print(d)

if __name__ == "__main__":
    main()
