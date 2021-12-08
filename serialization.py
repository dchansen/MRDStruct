import numpy as np
import struct
import functools


def read(source, type):
    tmp = source.read(type.size)
    return type.unpack(tmp)[0]

def read_vector(source, np_type=np.uint64):
    size = read(source, uint64)
    dtype = np.dtype(np_type)
    return np.frombuffer(source.read(size * dtype.itemsize), dtype=dtype)

def read_array(source, np_type=np.uint64):
    dtype = np.dtype(np_type)
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(np.frombuffer(source.read(elements * dtype.itemsize), dtype=dtype), dimensions, order='F')

def read_byte_string(source):
    length = read(source,uint64)
    byte_string = source.read(length)
    return byte_string

def read_type(source):
    return typemap[read(source, uint16)]

def read_typed_array(source):
    obj_type = read_type(source)
    return read_array(source, obj_type)

Bool = struct.Struct('<?')
uint32 = struct.Struct('<I')
uint64 = struct.Struct('<Q')
uint16 = struct.Struct('<H')

typemap = {
    1: np.dtype('float32'),
    2: np.dtype('float64'),
    3: np.dtype('uint16'),
    4: np.dtype('uint32'),
    5: np.dtype('uint64'),
    6: np.dtype('int16'),
    7: np.dtype('int32'),
    8: np.dtype('int64'),
    9: np.dtype('complex64'),
    10: np.dtype('complex128')
}

indexmap = { val: key for key,val in typemap.items()}

def write_vector(destination, values, type=uint64):
    destination.write(uint64.pack(len(values)))
    for val in values:
        destination.write(type.pack(val))


def write_array(destination, array, dtype):
    write_vector(destination, array.shape)
    array_view = np.array(array,dtype=dtype,copy=False)
    destination.write(array_view.tobytes(order='F'))


def write_byte_string(destination, byte_string):
    destination.write(uint64.pack(len(byte_string)))
    destination.write(byte_string)

def write_type(destination, type):
    destination.write(uint16.pack(indexmap[type]))


def write_typed_array(source,arr):
    write_type(source,arr.dtype)
    return write_array(source, arr, arr.dtype)