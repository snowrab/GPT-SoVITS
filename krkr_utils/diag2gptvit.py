import argparse
import os
import random
import shutil
import json
import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前日期和时间
current_time = datetime.datetime.now()

# 格式化日期时间字符串
# 例如："20240505_151230" 表示2024年5月5日15时12分30秒
time_str = current_time.strftime('%Y%m%d_%H%M%S')

cv2model = {
    "乃愛":"noa",
    "天音":"ama",
    "来海":"kur"
}

def read_file_to_list(filename):
    if filename == "":
        filename = os.path.join(script_dir, '../processed/texts/tenshi_normal.txt')
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # 将Python字典样式的字符串转换为JSON格式
            line = line.replace("'", '"').replace("None", "null")
            data.append(json.loads(line))
    return data


def filter_and_sample(data, k=0, n=1, cv=None, random_number=1, sample = True):
    random.seed(random_number)
    if cv:
        filtered_data = [item for item in data if item['voice'] is not None and len(item['content']) >= k and item['user']==cv]
    else:
        filtered_data = [item for item in data if item['voice'] is not None and len(item['content']) >= k]
    user_dict = {}

    for item in filtered_data:
        user_dict.setdefault(item['user'], []).append(item)

    sampled_data = []
    if sample:
        for user, items in user_dict.items():
            sampled_items = random.sample(items, min(len(items), n))  # 防止n大于用户字典长度
            sampled_data.extend(sampled_items)
    else:
        for user, items in user_dict.items():
            sampled_data.extend(items)

    return sampled_data

def write_to_file(sampled_data, filename):
    with open(filename, 'w') as file:
        for item in sampled_data:
            model_name = os.path.split(item['voice'])[-1][:3]
            line = f"{item['voice']}|{model_name}|JP|{item['content']}\n"
            file.write(line)

def filter_sample_and_copy(data, k, n, random_number,voice_dir,cv,outdir):
    if voice_dir == "":
        voice_dir = os.path.join(script_dir, '../data/tenshi/voices/')
    os.makedirs(outdir, exist_ok=True)
    dir1 = os.path.join(outdir,"raw")
    os.makedirs(dir1, exist_ok=True)

    sampled_data = filter_and_sample(data, k, n, cv=cv, random_number=random_number)

    write_to_file(sampled_data, os.path.join(outdir,"esd.list"))

    for item in sampled_data:
        voice_file = os.path.join(voice_dir,item['voice'])

        shutil.copy(voice_file, dir1)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理生成Bert-vits2模型的输入数据")
    parser.add_argument('-sp', '--scene_path', type=str, default="", help="scene文本路径")
    parser.add_argument('-r', '--random_seed', type=int, default=1, help="随机种子")
    parser.add_argument('-l', '--min_length', type=int, default=20, help='最小对话长度')
    parser.add_argument('-s', '--size', type=int, default=40, help='采样数')
    parser.add_argument('-vd', '--voice_dir', type=str, default="", help="wav文件夹路径")
    parser.add_argument('--cv', type=str, default="乃愛", help="参考音频角色")
    parser.add_argument('--out', type=str, default='../processed/data/', help='The path to store wav files and their list.')
    parser.add_argument("-n", "--name", type=str, default="", help="转化模式：只将scene格式化为list")
    args = parser.parse_args()

    
    base_outdir = os.path.join(script_dir, args.out)
    outdir = os.path.join(base_outdir, time_str)
    os.makedirs(outdir, exist_ok=True)



    data = read_file_to_list(args.scene_path)
    print(type(data[0]))
    if args.name:
        all_data = filter_and_sample(data,sample=False)
        file_path = os.path.join(outdir,args.name+".list")
        write_to_file(all_data,file_path)
    else:
        filter_sample_and_copy(data, args.min_length, args.size, args.random_seed,args.voice_dir, cv = args.cv, outdir = outdir)