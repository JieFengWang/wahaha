rm -rf build 
mkdir build 
cd build; 
cmake .. 
make -j 
cd ..

# rm output.bin
./build/test 

## 1m
# ./build/test /home/jfwang/proj/SSS/get_data/contest-data-release-1m.bin /home/jfwang/proj/SSS/get_data/contest-queries-release-1m.bin ggt1m_test_eva.bin

# # ##10m

# ./build/test /media/media01/jfwang/proj/SSS/data/a.bin /media/media01/jfwang/proj/SSS/data/b.bin /media/media01/jfwang/proj/SSS/data/out-10m_new.bin
