
if __name__=='__main__':
    import sys
    import h5py
    import numpy as np
    import os

    hdf5_file=sys.argv[1]

    output_npy=hdf5_file+'.npy'
    f = h5py.File(hdf5_file, 'r')
    np_dict={}
    for key in f:
        np_dict[key]=f[key][:]
        # print (key)
    f.close()

    np.save(output_npy,np_dict)
    os.remove(hdf5_file)
    print (hdf5_file, 'removed')
    # with open(output_json,'w') as f:
    #     json.dump(f_json,f)

