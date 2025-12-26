#!/bin/bash

# --- 配置部分 ---
NCU_INSTALL_DIR="$HOME/ncu"
SOURCE_FILE="bench"      # 你的源文件名（不带后缀）
OUTPUT_EXE="bench_exe"   # 编译后的可执行文件名

echo ">>> 1. 正在检查并安装 Nsight Compute..."
# 查找当前目录下的 .run 安装包
INSTALLER=$(ls nsight_compute-linux-*.run 2>/dev/null | head -n 1)

if [ -z "$INSTALLER" ]; then
    echo "未找到安装包，跳过安装步骤（假设已安装）"
else
    # 使用之前调试通过的正确参数：--quiet -- -noprompt
    echo "发现安装包: $INSTALLER，正在安装到 $NCU_INSTALL_DIR ..."
    sh "$INSTALLER" --quiet -- -noprompt -targetpath="$NCU_INSTALL_DIR"
    echo "安装完成。"
fi

echo ">>> 2. 配置环境变量..."
# 添加 CUDA 编译器 (nvcc) 路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 添加 Nsight Compute (ncu) 路径
# 注意：ncu 的可执行文件通常在安装目录下的 target/linux-desktop-glibc_... 文件夹里
# 这里使用 find 自动查找 ncu 可执行文件的位置
NCU_BIN=$(find "$NCU_INSTALL_DIR" -name ncu -type f | head -n 1)
if [ -n "$NCU_BIN" ]; then
    NCU_PATH=$(dirname "$NCU_BIN")
    export PATH=$NCU_PATH:$PATH
    echo "已添加 ncu 路径: $NCU_PATH"
else
    echo "警告：未找到 ncu 可执行文件，稍后可能会报错。"
fi

echo ">>> 3. 编译 CUDA 代码..."
# 自动处理没有 .cu 后缀的问题
if [ -f "$SOURCE_FILE" ] && [ ! -f "$SOURCE_FILE.cu" ]; then
    echo "检测到源文件缺少后缀，正在重命名: $SOURCE_FILE -> $SOURCE_FILE.cu"
    mv "$SOURCE_FILE" "$SOURCE_FILE.cu"
fi

# 编译 (使用 -arch=native 优化性能)
if [ -f "$SOURCE_FILE.cu" ]; then
    nvcc -o "$OUTPUT_EXE" "$SOURCE_FILE.cu" -arch=native
    echo "编译成功: $OUTPUT_EXE"
else
    echo "错误：找不到源代码文件 $SOURCE_FILE.cu"
    exit 1
fi

echo ">>> 4. 开始性能分析..."
echo "尝试使用 Nsight Compute (ncu)..."

# 尝试运行 ncu
ncu -o profile_result ./$OUTPUT_EXE

# 检查 ncu 的退出状态
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "--------------------------------------------------------"
    echo "检测到 ncu 运行失败 (可能是 AutoDL 权限限制 ERR_NVGPUCTRPERM)。"
    echo "正在自动切换为 Nsight Systems (nsys)，这个通常能在 AutoDL 上运行。"
    echo "--------------------------------------------------------"
    
    # Fallback 到 nsys
    nsys profile -o profile_result --stats=true ./$OUTPUT_EXE
    
    echo ">>> 完成。如果 nsys 成功，请下载生成的 .nsys-rep 文件到本地查看。"
else
    echo ">>> ncu 运行成功！请下载生成的 .ncu-rep 文件。"
fi