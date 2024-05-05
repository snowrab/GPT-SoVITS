import os
import json
import argparse



def read_json_files(directory, prefix):
    json_data = []
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r') as file:
                data = json.load(file)
                json_data.append(data)
    
    return json_data

def convert_json2diaglog(prefix,directory='.',notsplit=False,output=True,outdir='.'):
    data_list = read_json_files(directory,prefix)
    normal,hscene = [],[]
    split = not notsplit
    for data in data_list:
        for scene in data["scenes"]:
            if "texts" in scene.keys():
                for text in scene["texts"]:
                    diag = {}
                    diag["user"] = text[0]
                    diag["content"] = text[1][0][1].replace("「", "").replace("」", "")
                    if text[2]:
                        diag["voice"] = text[2][0]['voice'].split("|")[0].split(".")[0]+".wav"
                    else:
                        diag["voice"] = None
                    is_hscene = text[4]["_meswinchange"]=="hscene"
                    if split and is_hscene:
                        hscene.append(diag)
                    else:
                        normal.append(diag)
    if output:
        normaltxt = os.path.join(outdir,prefix+"_normal.txt")
        hscenetxt = os.path.join(outdir,prefix+"_hscene.txt")
        with open(normaltxt, 'w') as file:
            for item in normal:
                file.write(str(item) + '\n')
        if split:
            with open(hscenetxt, 'w') as file:
                for item in hscene:
                    file.write(str(item) + '\n')

    return normal,hscene

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='生成对话的txt文件')
    parser.add_argument('input', type=str, help='输入json前缀')
    parser.add_argument('--notsplit', action='store_true', help='不区分不同的scene')
    parser.add_argument('--jsonpath', type=str, default='scns/', help='The path to read json files.')
    parser.add_argument('--out', type=str, default='../processed/texts/', help='The path to store txt results.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, args.out)
    os.makedirs(outdir, exist_ok=True)

    normal,hscene = convert_json2diaglog(prefix=args.input,directory=args.jsonpath,notsplit=args.notsplit,output=True,outdir=outdir)