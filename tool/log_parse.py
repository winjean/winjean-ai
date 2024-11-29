import re
import yaml

# 定义日志解析的正则表达式
log_pattern = re.compile(r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d{3}) (\d+|-)')


# 读取日志文件
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


# 解析日志行
def parse_log_line(line):
    match = log_pattern.match(line)
    if match:
        ip, identity, user, timestamp, request, status, size = match.groups()
        return {
            'ip': ip,
            'identity': identity,
            'user': user,
            'timestamp': timestamp,
            'request': request,
            'status': int(status),
            'size': int(size) if size.isdigit() else None
        }
    return None


# 解析整个日志文件
def parse_log_file(file_path):
    lines = read_log_file(file_path)
    parsed_logs = []
    for line in lines:
        parsed_log = parse_log_line(line)
        if parsed_log:
            parsed_logs.append(parsed_log)
    return parsed_logs


# 将解析后的日志转换成 YAML 格式
def logs_to_yaml(parsed_logs):
    return yaml.dump(parsed_logs, default_flow_style=False)


# 主函数
def main():
    log_file_path = 'error.log'
    parsed_logs = parse_log_file(log_file_path)
    yaml_output = logs_to_yaml(parsed_logs)
    print(yaml_output)

    # 保存到文件
    with open('parsed_logs.yaml', 'w') as file:
        file.write(yaml_output)


if __name__ == '__main__':
    main()

