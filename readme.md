# 赋予执行权限 （如果已有ncu则跳过）
```bash
chmod +x nsight-compute-linux-*.run
```
# 安装ncu
```bash
./nsight_compute-linux-*.run --quiet -- -noprompt -targetpath="$HOME/ncu"
```
# 添加ncu到Path （如果可在终端运行ncu则跳过）
```bash
export PATH=$HOME/ncu/target/linux-desktop-glibc_2_11_3-x64:$PATH
```
# 添加nvcc到Path（如果可在终端运行nvcc则跳过）
```bash
PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

# 编译并运行7_1 MSE kernel
```bash
cd 7_1
python gen.py
nvcc -o bench bench.cu
./bench
python compare.py
ncu -o bench ./bench
```

# 运行
```bash
cd ..
python ./test.py 7_1 bench  #因为7_1的bench已经编译完成，所以参数直接输入bench 
python ./test.py 7_1_op     #反之则使用默认参数，会自动编译
python ./test.py 5_1
```

# 如果ncu不可用
```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiler.conf
sudo reboot # 重启 
```