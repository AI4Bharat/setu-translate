import os

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather,host_local_array_to_global_array
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P

# own_jax_setup(cpu_devices=2 * 2)


jax.distributed.initialize()
jax.config.update("jax_enable_x64", True)

print(jax.devices(), jax.local_devices())
print("process_index", jax.process_index())
devices = mesh_utils.create_device_mesh((2, 2))

mesh = Mesh(devices, axis_names=('a', 'b'))

size = 15


@pjit
def func(x):
    return jax.numpy.sqrt(x)


with mesh:
    assert jax.process_count() == 2
    local_shape = (2 ** (size - 1), 2 ** size)

    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
    print("global_shape", global_shape)

    if jax.process_index() == 0:
        dataX = jax.numpy.arange(2 ** (size - 1))
    else:
        dataX = jax.numpy.arange(2 ** (size - 1), 2 ** size)

    dataY = jax.numpy.arange(2 ** size)
    local_array = jax.numpy.outer(dataX, dataY)
    print("local_array shape", local_array.shape)
    assert local_array.shape == local_shape

    split = jax.numpy.split(local_array, len(mesh.local_devices), axis=0)
    print(split[0].shape)
    arrays = jax.device_put(
        split,
        mesh.local_devices
    )

    sharding = jax.sharding.NamedSharding(mesh, P(('a', 'b'), ))
    x = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

    z = func(x)
    jax.debug.visualize_array_sharding(z)
    jax.debug.visualize_array_sharding(x)

    result = np.asarray(process_allgather(z))
    print(result.shape)
    start = np.asarray(process_allgather(x))
    assert np.allclose(np.sqrt(start), result)
    print(result.size)