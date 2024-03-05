import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 日志文件路径
log_file_path = './result/log.txt'

# 从日志中提取时间
timestamps = []
with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Find the end position of date_time in the line
        end_pos = line.find('小车最终落于[')
        date_time_str = line[:end_pos].strip()
        # Assume all logs are in the same year
        date_time_str = '2023-' + date_time_str
        timestamps.append(datetime.strptime(date_time_str, '%Y-%m-%d %H:%M'))

# Calculate run times for each program run and store them in a list named run_times
run_times = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]

# Get the index and value of max and min run times
max_index = run_times.index(max(run_times))
min_index = run_times.index(min(run_times))

# Calculate average run time
avg_run_time = np.mean(run_times)

# Draw the line chart
plt.plot(run_times)
plt.scatter([max_index], [max(run_times)], color='r')  # Red color for max run time
plt.scatter([min_index], [min(run_times)], color='g')  # Green color for min run time
plt.axhline(y=avg_run_time, color='b', linestyle='--')  # Blue color for average run time
plt.text(max_index, max(run_times), f'Max: {max(run_times)}s', fontsize=9, verticalalignment='bottom')
plt.text(min_index, min(run_times), f'Min: {min(run_times)}s', fontsize=9, verticalalignment='top')
plt.text(len(run_times)-1, avg_run_time+5, f'Avg: {avg_run_time:.2f}s', fontsize=9, horizontalalignment='right')
plt.xlabel('Run number')
plt.ylabel('Run time (seconds)')
plt.savefig('run_times.png')
plt.show()
