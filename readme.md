# 赋予执行权限
chmod +x nsight-compute-linux-*.run

# 安装
./nsight_compute-linux-*.run --quiet -- -noprompt -targetpath="$HOME/ncu"

# 添加到Path
export PATH=$HOME/ncu/target/linux-desktop-glibc_2_11_3-x64:$PATH

# nvcc
PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 尝试运行
cd 7_1_op
python gen.py
nvcc -o bench bench.cu
./bench
python compare.py
ncu -o bench ./bench

# 运行
cd ..
python ./test.py 7_1_op bench
python ./test.py 7_1
python ./test.py 5_1

# 
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiler.conf
sudo reboot