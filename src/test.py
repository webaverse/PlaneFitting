# say we have a set of planes each represented as an ndarray of 3d points
# we want to call ./PlaneDetection.exe to compute the plane coefficients for each plane
# pass the planes to PlaneDetection.exe a serialized binary, and parse the pure binary normalized plane coefficients from the output

import subprocess
import numpy as np
from pprint import pprint
import struct

# // use node to output the binary float reprentation of [1,2,3,2,3,4,3,4,5,4,5,6,5,6,7] to stdout for testing...
# // node -e 'process.stdout.write(new Uint8Array(new Float32Array([1,2,3,2,3,4,3,4,5,4,5,6,5,6,7]).buffer))'
# same thing in python, serialize the struct to bytes:
points = struct.pack('f'*15, 1,2,3,2,3,4,3,4,5,4,5,6,5,6,7)
# points = np.array([np.random.rand(100, 3), np.random.rand(100, 3), np.random.rand(100, 3)])
result = subprocess.run(["./PlaneFittingSample"], input=points, stdout=subprocess.PIPE)
planes = np.frombuffer(result.stdout, dtype=np.float32)
# print the plane
pprint(planes.reshape(-1, 7))