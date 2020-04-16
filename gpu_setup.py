# -*- coding: utf-8 -*-

import tensorflow as tf

def gpu_setup(use_gpu):
    if use_gpu >= 0:
        '''
        Tensorflowはほっとくと見えるGPUのリソースを全部確保しようとするので
        (特に共用PCの上では)制限を加えてやる
        
        > 1人1台のGPUがあるならset_memory_growthの行はコメントアウトでもよい
        '''
        # List of GPUs
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        try:
            # Only grow the memory usage as is needed by the process
            # set_memory_growthは全GPUに適用しないと
            #       Memory growth cannot differ between GPU devices
            # のエラーが出る
            # for gpu in physical_devices:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # Use single GPU #<use_gpu>
            tf.config.experimental.set_visible_devices(physical_devices[use_gpu], 'GPU')
    
        except RuntimeError as e:
            print(e)
        
        print('Using GPU: {}'.format(tf.config.experimental.list_logical_devices('GPU')[0].name))
        return tf.config.experimental.list_logical_devices('GPU')[0].name
    
    else:
        print('Using CPU: /CPU:0')
        return '/CPU:0'

if __name__ == '__main__':
    # Set execution device
    use_gpu = 0 # if use CPU := -1
    
    device = gpu_setup(use_gpu)
   
    print('Visible GPU Devices: {}'.format(tf.config.experimental.get_visible_devices('GPU')))
    print('Logical Devices: {}'.format(tf.config.experimental.list_logical_devices('GPU')))

    with tf.device(device): # GPUで演算    
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    
    print(c.device)