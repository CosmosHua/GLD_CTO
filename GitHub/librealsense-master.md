https://pypi.org/project/pyrealsense2

https://pypi.org/project/pyrealsense2-aarch64

https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

> pip3 install pyrealsense2
>pip3 install pyrealsense2-aarch64

#### Ubuntu 16.04/18.04 LTS

1. Ensure apt-get is up to date

- `sudo apt-get update && sudo apt-get upgrade`

2. Install Python and its development files via apt-get (Python 2 and 3 both work)

- `sudo apt-get install python3 python3-dev`

3. Run the top level CMake command with the following additional flag `-DBUILD_PYTHON_BINDINGS=true`:

- `git clone https://github.com/IntelRealSense/librealsense.git`
- `cd librealsense`

> **Note**: Building python wrapper from source [requires invoking the CMake from the topmost](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#building-from-source) (i.e. Librealsense's root) directory. [#4169](https://github.com/IntelRealSense/librealsense/issues/4169)

- `mkdir build`
- `cd build`
- `cmake ../ -DBUILD_PYTHON_BINDINGS=true -DPYTHON_EXECUTABLE=$(which python3)`

> **Note**: To force compilation with a specific version on a system with both Python 2 and Python 3 installed, add the following flag to CMake command: `-DPYTHON_EXECUTABLE=[full path to the exact python executable]`

- `make -j4`
- `sudo make install`

4. update your PYTHONPATH environment variable to add the path to the pyrealsense library

- `gedit ~/.bashrc`
  - `export PYTHONPATH=$PYTHONPATH:/usr/local/lib`
  - `export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2`

5. Alternatively, copy the build output (`librealsense2.so` and `pyrealsense2.so`) next to your script.

> **Note:** Python 3 module filenames may contain additional information, e.g. `pyrealsense2.cpython-35m-arm-linux-gnueabihf.so`)