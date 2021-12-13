import ismrmrd
import numpy as np
import struct
import functools
import ctypes
import dataclasses

def read_value(source, type):
    tmp = source.read(type.size)
    return type.unpack(tmp)[0]


def read_vector(source, np_type=np.uint64):
    size = read_value(source, uint64)
    dtype = np.dtype(np_type)
    return np.frombuffer(source.read(size * dtype.itemsize), dtype=dtype)


def read_cstruct(source, ctype):
    return ctype.from_buffer_copy(source.read(ctypes.sizeof(ctype)))


def read_array(source, data_type):
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    dtype = np.dtype(data_type)

    if dtype == object:
        return np.reshape(np.array([read_cstruct(source, data_type) for i in range(elements)]), dimensions, order='F')

    return np.reshape(np.frombuffer(source.read(elements * dtype.itemsize), dtype=dtype), dimensions, order='F')


def read_byte_string(source):
    length = read_value(source, uint64)
    byte_string = source.read(length)
    return byte_string


def read_type(source):
    return typemap[read_value(source, uint16)]


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
    10: np.dtype('complex128'),
    11: ismrmrd.AcquisitionHeader,
    12: ismrmrd.ImageHeader,
    13: ismrmrd.WaveformHeader
}

indexmap = {val: key for key, val in typemap.items()}


def write_vector(destination, values, type=uint64):
    destination.write(uint64.pack(len(values)))
    for val in values:
        destination.write(type.pack(val))


def __writer_array_content(destination, array, data_type):
    dtype = np.dtype(data_type)
    if (dtype == object or not dtype.isbuiltin) and issubclass( data_type, ctypes.Structure):
        for item in np.nditer(array, ('refs_ok', 'zerosize_ok'), order='F'):
            item = item.item()  # Get rid of the numpy 0-dimensional array.
            destination.write(item)
    else:
        array_view = np.array(array, dtype=dtype, copy=False)
        destination.write(array_view.tobytes(order='F'))


def write_array(destination, array, data_type):
    write_vector(destination, array.shape)
    __writer_array_content(destination, array, data_type)


def write_byte_string(destination, byte_string):
    destination.write(uint64.pack(len(byte_string)))
    destination.write(byte_string)


def write_type(destination, type):
    destination.write(uint16.pack(indexmap[type]))


def type_from_array(arr):
    if arr.dtype == object and len(arr) > 0:
        return type(arr[0])
    return arr.dtype


def write_typed_array(source, arr):
    write_type(source, type_from_array(arr))
    return write_array(source, arr, type_from_array(arr))

def from_dict(obj_dict, datatype):

	return datatype(*(load_field(obj_dict,field) for field in dataclasses.fields(datatype)))


def load_field(obj_dict,field):
	if dataclasses.is_dataclass(field.type):
		return from_dict(obj_dict[field.name],field.type)
	return obj_dict[field.name]
	