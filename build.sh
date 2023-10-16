mkdir build # create build directory
cd build # enter build directory
cmake .. # build libigl project (automatically load external libs)
#$ build/ cmake .. -G "Visual Studio 15 2017 Win64" # for window
make # build libraries

#build/ cmake --build .