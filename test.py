import os
import subprocess
import uuid
import pandas as pd
import argparse
import sys

import os
import subprocess
import pandas as pd
import argparse

def eval_eff(executable_path, draw_flag, label=""):
    metric = [
        # peak
        "dram__bytes.sum.peak_sustained",
        "dram__cycles_elapsed.avg.per_second",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
        "sm__cycles_elapsed.avg.per_second",
        # achieved
        "dram__bytes.sum.per_second",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed",
        "smsp__cycles_elapsed.avg.per_second"
        # "sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
    ]
    metrics = ",".join(metric)

    csv_output_folder = './tmp_csv/'
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)
    
    # ================= 修改 1：生成固定的文件名 =================
    # 获取可执行文件的名字（去除路径），例如 "./main" -> "main"
    exe_name = os.path.basename(executable_path)
    # 构造文件名：Label + 可执行文件名.csv (去掉了随机 UUID)
    csv_filename = f"{label}{exe_name}.csv"
    csv_output_path = os.path.join(csv_output_folder, csv_filename)

    # ================= 修改 2：检查文件是否存在 =================
    # 检查文件是否存在，且大小大于0（避免空文件）
    file_exists = os.path.exists(csv_output_path) and os.path.getsize(csv_output_path) > 0

    if file_exists:
        print(f"[Info] Found existing CSV: {csv_output_path}. Skipping NCU execution.")
    else:
        print(f"[Info] No existing CSV found. Running NCU for: {exe_name}")
        # 执行 ncu 性能分析命令
        ncu_command = ['ncu', '--metrics', metrics, '--csv', executable_path]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        try:
            with open(csv_output_path, 'w') as output_file:
                # 注意：去掉了 ncu_command 中的 '>'，通过 stdout 参数处理输出
                subprocess.run(ncu_command, env=env, check=True, stdout=output_file, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[Error] NCU failed: {e}")
            if e.stderr:
                print(e.stderr.decode())
            return None

    # ================= 后续的数据处理逻辑 (保持不变) =================
    # 评估效率
    try:
        with open(csv_output_path, 'r') as f:
            cnt=0
            while True:
                ln=f.readline()
                if not ln:
                    break
                cnt+=1
                if 'Host Name' in ln:
                    break
        
        if cnt == 0: 
            print("[Error] CSV file is empty or invalid.")
            return None

        df = pd.read_csv(csv_output_path, skiprows=cnt-1)
        df['Metric Value'] = df['Metric Value'].replace({',': ''}, regex=True).astype(float)
        
        dft=df.groupby(['Kernel Name','Metric Name']).sum()
        dfmetric=pd.pivot_table(dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
        dfmetric['Count']=df.groupby(['Kernel Name']).count()['ID'].div(dfmetric.shape[1])

        dfmetric['Peak S FLOPs']= 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'] / dfmetric['Count']
        dfmetric['PeakWork']= dfmetric['Peak S FLOPs'] * dfmetric['sm__cycles_elapsed.avg.per_second'] / dfmetric['Count']
        dfmetric['PeakTraffic']= dfmetric['dram__bytes.sum.peak_sustained'].div(dfmetric['Count']) * dfmetric['dram__cycles_elapsed.avg.per_second'].div(dfmetric['Count'])

        dfmetric['S FLOPs']= 2 * dfmetric['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'] \
                            + dfmetric['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed']
        dfmetric['all FLOPs']= (dfmetric['S FLOPs']) / (dfmetric['Count'])
        
        # 原始计算
        raw_flops_per_sec = dfmetric['all FLOPs'] * dfmetric['smsp__cycles_elapsed.avg.per_second'].div(dfmetric['Count'])
        
        # 单位换算：GFlops/s
        dfmetric['GFLOP/s'] = raw_flops_per_sec / 1e9

        dfmetric['AI DRAM'] = raw_flops_per_sec.div(dfmetric['dram__bytes.sum.per_second'].div(dfmetric['Count']))

        flops = dfmetric['GFLOP/s'].item()
        peak_work = dfmetric['PeakWork'].item() / 1e9
        ai_dram = dfmetric['AI DRAM'].item()
        peak_traffic = dfmetric['PeakTraffic'].item()

        bandwidth_utilization = (
            dfmetric['dram__bytes.sum.per_second']
            .div(dfmetric['Count'])
            .item()
            / dfmetric['PeakTraffic'].item()
        )


        EPSILON = 1e-9

        if abs(flops) < EPSILON:
            compute_efficiency = 0.0
            score = bandwidth_utilization
        else:
            compute_efficiency = flops / peak_work
            roofline_limit = min(peak_work, ai_dram * peak_traffic / 1e9)
            score = flops / roofline_limit

        return flops, roofline_limit, ai_dram, peak_traffic, score
        
    except Exception as e:
        print(f"[Error] Failed to process CSV: {e}")
        return None

# --- 修改后的核心逻辑 ---
def run_test_in_folder(target_folder, executable_name=None):
    """
    切换到 target_folder。
    如果 executable_name 为 None：
      1. 寻找 bench.cu 并编译。
      2. 寻找 gen.py 并运行。
      3. 将 executable_name 设为 "bench"。
    然后运行 eval_eff。
    最后切回原目录。
    """
    abs_target_folder = os.path.abspath(target_folder)
    original_cwd = os.getcwd()
    
    # 检查文件夹是否存在
    if not os.path.exists(abs_target_folder):
        print(f"[Error] Target folder '{abs_target_folder}' does not exist.")
        return

    print(f"--- Context Switch ---")
    print(f"Original Dir: {original_cwd}")
    print(f"Target Dir  : {abs_target_folder}")

    try:
        # 1. 切换工作目录
        os.chdir(abs_target_folder)
        
        # 2. 自动构建逻辑 (如果未指定执行文件)
        if executable_name is None:
            print("\n[Auto-Build] No executable specified. Looking for 'bench.cu'...")
            
            # 检查 bench.cu 是否存在
            if os.path.exists("bench.cu"):
                # 步骤 A: 编译 bench.cu
                compile_cmd = ["nvcc", "-o", "bench", "bench.cu"]
                print(f"[Compile] Executing: {' '.join(compile_cmd)}")
                try:
                    subprocess.run(compile_cmd, check=True)
                except subprocess.CalledProcessError:
                    print("[Error] Compilation failed (nvcc). Exiting.")
                    return

                # 步骤 B: 生成测试数据 (如果有 gen.py)
                if os.path.exists("gen.py"):
                    gen_cmd = ["python", "gen.py"]
                    print(f"[Data Gen] Executing: {' '.join(gen_cmd)}")
                    try:
                        subprocess.run(gen_cmd, check=True)
                    except subprocess.CalledProcessError:
                        print("[Error] Data generation failed (gen.py). Exiting.")
                        return
                else:
                    print("[Warn] 'gen.py' not found. Skipping data generation.")
                
                # 设定接下来的执行目标为刚刚编译出的 bench
                executable_name = "bench"
                
            else:
                print("[Error] 'bench.cu' not found in this directory. Cannot auto-build.")
                return

        # 3. 处理执行文件路径前缀
        if not executable_name.startswith('./') and not executable_name.startswith('/'):
            run_cmd = f"./{executable_name}"
        else:
            run_cmd = executable_name

        # 检查最终要运行的文件是否存在
        if not os.path.isfile(executable_name):
             print(f"[Error] Executable '{executable_name}' not found in {abs_target_folder}")
             return

        print(f"\n[Execute] Target: {run_cmd}")
        
        # 4. 调用 eval_eff 进行性能分析
        result = eval_eff(run_cmd, draw_flag=False, label="auto_")
        
        if result:
            flops, peak_work, ai, peak_traffic, score = result
            print(f"\n=== Result for {executable_name} ===")
            print(f"FLOP/s      : {flops:.4e} GFlops/s")
            print(f"Peak FLOP/s : {peak_work:.4e} GFlops/s")
            print(f"AI DRAM     : {ai:.4f}")
            print(f"Peak Traffic: {peak_traffic:.4f}")
            print(f"Score       : {score:.4f}")
        else:
            print("\n=== Result: Failed to calculate metrics ===")

    except OSError as e:
        print(f"[OS Error] {e}")
    finally:
        # 5. 切回原目录
        os.chdir(original_cwd)
        print(f"\n--- Restored Dir: {os.getcwd()} ---")

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCU Roofline test.")
    
    # 参数 1: 目标文件夹 (必须)
    parser.add_argument("folder", help="The folder containing the code/data.")
    
    # 参数 2: 可执行文件名 (可选)
    parser.add_argument("executable", nargs='?', help="Executable name. If omitted, attempts to compile 'bench.cu' and run 'gen.py'.")
    
    args = parser.parse_args()
    
    run_test_in_folder(args.folder, args.executable)