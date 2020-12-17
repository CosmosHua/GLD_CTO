https://github.com/ros-perception/vision_opencv

#### ROS中使用Python3
> pip3 install catkin_pkg pyyaml empy rospkg numpy

#### 在Python3中调用cvbridge：
> https://blog.csdn.net/weixin_44060400/article/details/104347628
> https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3

> `python-catkin-tools` is needed for catkin tool
> `python3-dev` and `python3-catkin-pkg-modules` is needed to build cv_bridge
> `python3-numpy` and `python3-yaml` is cv_bridge dependencies
> `ros-kinetic-cv-bridge` (for ubuntu16) or `ros-melodic-cv-bridge` (for ubuntu18) is needed to install a lot of cv_bridge deps. Probaply you already have it installed.

`sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge`

> Create catkin workspace
```
mkdir catkin_ws
cd catkin_ws
catkin init
```
> Instruct catkin to set cmake variables：用系统python3.6能编译成功，用miniconda会部分编译失败

`catkin config -DPYTHON_EXECUTABLE=$(which python3.6) -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so`

> Instruct catkin to install built packages into install place. It is $CATKIN_WS/install folder

`catkin config --install`

> Clone `cv_bridge` src

`git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv`

> Find version of cv_bridge in your repository

```
apt-cache show ros-melodic-cv-bridge | grep Version
> Version: 1.13.0-0bionic.20200530.112157
```

> Checkout right version in git repo. In our case it is 1.13.0
```
cd src/vision_opencv/
git checkout 1.13.0
cd ../../
```
> Build

`catkin build cv_bridge`

> Extend environment with new package
```
echo $(pwd)/install/setup.bash --extend > ~/.bashrc
source install/setup.bash --extend
python3
> from cv_bridge.boost.cv_bridge_boost import getCvType
```

