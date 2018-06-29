import argparse
import h5py
import numpy as np


def convert_weights(weight_file, output_weights, dtype):
    weights = h5py.File(weight_file, mode='r')
    qweights = h5py.File(output_weights, mode='w')
    try:
        layers = weights.attrs['layer_names']
    except:
        raise ValueError("weights file must contain attribution: 'layer_names'")
    qweights.attrs['layer_names'] = [name for name in weights.attrs['layer_names']]
    for layer_name in layers:
        f = qweights.create_group(layer_name)
        g = weights[layer_name]
        f.attrs['weight_names'] = g.attrs['weight_names']
        for weight_name in g.attrs['weight_names']:
            print(weight_name)
            weight_value = g[weight_name].value
            weight_value = weight_value.astype(dtype)
            param_dest = f.create_dataset(weight_name, weight_value.shape, dtype=dtype)
            param_dest[:] = weight_value
    qweights.flush()
    qweights.close()
    print('Converting done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='command for converting keras pre-trained weights')
    parser.add_argument('--input-weights', type=str, default='./weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    parser.add_argument('--output-weights', type=str, default='./weights/inception_resnet_v2_tf_fp16.h5')
    parser.add_argument('--dtype', type=str, default='float16')
    args = parser.parse_args()

    if args.dtype == 'float16':
        dtype = np.float16
    else:
        dtype = np.float32

    convert_weights(args.input_weights, args.output_weights, dtype)
