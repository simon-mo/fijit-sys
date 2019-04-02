import subprocess
import os
import pandas as pd
import numpy as np
from io import StringIO
from collections import namedtuple

PLOT_DST = "/home/ubuntu/plots"
FIJIT_ROOT = "/home/ubuntu/code/fijit/"
FIJIT_BIN = os.path.join(FIJIT_ROOT, "build", "qps_bench")
UTIL_CMD = ["/usr/bin/nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory", "--format=csv", "-lms", "1"]

BenchResult = namedtuple('BenchResult', 'num_streams, max_blocks query_id time_ms')

class BenchTask(object):
    def bench(self):
        raise NotImplementedError

class BenchFijit(BenchTask):
    def __init__(self, qps=1, num_streams=1, max_blocks=100000, num_query=128, burst_batch_size=1, model=None, input_proto=None, input_name=None):
        self.args = {
            'model': model or os.path.join(FIJIT_ROOT, "data/resnet50.onnx"),
            'input_proto': input_proto or os.path.join(FIJIT_ROOT, "data/resnet50_input.onnx"),
            'input_name': input_name or '0'
        }

        self.params = { # num_bursts, num_query_burst, inter_burst_ms
            'num_streams': num_streams,
            'max_blocks': max_blocks,
            'num_queries_total': num_query,
            'num_query_burst': burst_batch_size,
            'num_bursts': num_query // burst_batch_size,
            'inter_burst_ms': int(burst_batch_size * (1000 / qps)),
        }

        self.cmd = " ".join(str(x) for x in [
            FIJIT_BIN,
            "--model", self.args.get('model'),
            "--input", self.args.get('input_proto'),
            "--input-name", self.args.get('input_name'),
            "--num-stream", self.params.get('num_streams'),
            "--max-block", self.params.get('max_blocks'),

            "--num-bursts", self.params.get('num_bursts'),
            "--num-query-burst", self.params.get('num_query_burst'),
            "--inter-burst-ms", self.params.get('inter_burst_ms'),
        ])

        print(self.cmd)

    def bench(self):
        bench_out = subprocess.getoutput(self.cmd)
        return bench_out

def profile(bench_task: BenchTask):
    proc = subprocess.Popen(UTIL_CMD, stdout=subprocess.PIPE)
    bench_output = bench_task.bench()
    # print('---->\t'.join(bench_output.splitlines(True)))
    os.system('kill -15 ' + str(proc.pid))
    csv = proc.communicate()[0]

    # clean DF to start and end of benchmark (first/last non-zero values)
    df = pd.read_csv(StringIO(str(csv.decode())))
    df = df.rename(columns = {' utilization.gpu [%]': 'utilization.gpu', ' utilization.memory [%]': 'utilization.memory'})
    df['utilization.gpu'] = df['utilization.gpu'].apply(lambda x: int(x.split(' %', 1)[0]))
    df['utilization.memory'] = df['utilization.memory'].apply(lambda x: int(x.split(' %', 1)[0]))
    bench_start = min(df[['utilization.gpu', 'utilization.memory']].ne(0).idxmax())
    bench_end = max(df[['utilization.gpu', 'utilization.memory']].ne(0)[::-1].idxmax())
    df = df.iloc[bench_start:bench_end]
    df['timestamp'] = df['timestamp'].astype(str) + "000"
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]).astype('timedelta64[ms]')
    return df, bench_output

def burst_bench(qps, num_streams, max_blocks):
    task = BenchFijit(qps, num_streams, max_blocks, 128)
    df, bench_output = profile(task)
    ax = df.plot.line(x="timestamp", ylim=(0, 100))
    ax.get_figure().savefig(os.path.join(PLOT_DST, "ns{}_blocks{}.png".format(num_streams, max_blocks)))
    out_path = os.path.join(PLOT_DST, "ns{}_blocks{}.log".format(num_streams, max_blocks))
    with open(out_path, 'w') as f:
        f.write(bench_output)
    query_times_ms = [float(line.split(' ', 3)[2]) for line in bench_output.splitlines() if 'Total time: ' in line]
    if len(query_times_ms) == 0:
        raise Exception('No timings; cuda memory error?')
    # avg = np.mean(query_times_ms)
    # p50 = np.percentile(query_times_ms, 50)
    # p99 = np.percentile(query_times_ms, 99)
    return [BenchResult(num_streams, max_blocks, i, time_ms) for i, time_ms in enumerate(query_times_ms)]

if __name__ == "__main__":
    if not os.path.exists(PLOT_DST):
        os.makedirs(PLOT_DST)

    results = []
    for i in range(8):
        for qps in [1, 10, 100]:
            for ns in [1, 2, 4, 8]:
                for mb in [20, 40, 80]:
                    results.extend(burst_bench(qps, ns, mb))
    timing_df = pd.DataFrame.from_records(results, columns=BenchResult._fields)
    timing_df.to_csv(os.path.join(PLOT_DST, "results.csv"))
    print(timing_df)