# say we have a set of planes each represented as an ndarray of 3d points
# we want to call ./PlaneDetection.exe to compute the plane coefficients for each plane
# pass the planes to PlaneDetection.exe a serialized binary, and parse the pure binary normalized plane coefficients from the output

import subprocess
import numpy as np
from pprint import pprint
import struct

# // use node to output the binary float reprentation of [1,2,3,2,3,4,3,4,5,4,5,6,5,6,7] to stdout for testing...
# // node -e 'process.stdout.write(new Uint8Array(new Float32Array([1,2,3,2,3,4,3,4,5,4,5,6,5,6,7]).buffer))'
# same thing in python, serialize the struct to bytes in stdout:
# python3 -c 'import struct; import sys; sys.stdout.buffer.write(struct.pack("f"*15, 1,2,3,2,3,4,3,4,5,4,5,6,5,6,7))'
# create an array of 1000 points that are roughly at y=0 +- 0.01, and x,z range is [-100, 100]
points = np.random.rand(1024, 3) * 20 - 10
# set all y values to 0
points[:, 1] = 0
# add noise +- 0.01
points += np.random.rand(1024, 3) * 0.002 - 0.001
# points = np.array([np.random.rand(100, 3), np.random.rand(100, 3), np.random.rand(100, 3)])
# serialize the points to bytes
# pprint(points)

# add another plane above that, at 100m
points2 = np.random.rand(1024, 3) * 20 - 10
points2[:, 1] = 100
points2 += np.random.rand(1024, 3) * 0.002 - 0.001

points3 = np.concatenate([points, points2])

# shuffle the points
np.random.shuffle(points3)

points4 = struct.pack('f'*1024*3*2, *points3.flatten())
result = subprocess.run(["./PlaneFittingSample"], input=points4, stdout=subprocess.PIPE)
planes = np.frombuffer(result.stdout, dtype=np.float32)
# print the plane
pprint(planes.reshape(-1, 7))

# python3 -c 'import struct; import sys; sys.stdout.buffer.write(struct.pack("f"*15, 1,2,3,2,3,4,3,4,5,4,5,6,5,6,7))' | curl -X POST --data-binary @- https://depth.webaverse.com/ransac