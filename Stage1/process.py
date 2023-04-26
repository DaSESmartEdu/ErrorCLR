import graphviz
from graphviz import Graph, Digraph
import os
import json
from tqdm import tqdm
import networkx as nx
import random
import numpy as np
from infercode.client.infercode_client import InferCodeClient

infercode = InferCodeClient(language="c")
infercode.init_from_config()
node_type_dict = {}

def generate_graph(dir):
    with open("./load_json.sc", "w") as f:
        f.write("@main def exec() = {\n   loadCpg(\"cpg.bin\")\n   cpg.all.toJsonPretty |> \"output/" + dir + "/out.json\"\n}")
    os.system("./joern-parse ./input/" + dir + " && ./joern-export --repr cfg --out output/" + dir +" && ./joern --script load_json.sc")

def extract(dot_graph, func_name):
    for line in dot_graph.splitlines():
        if '(METHOD,' + func_name + ')' in line:
            id_ = line.split(" ")[0]
            break
    return id_


def construct_graph(graphs, func_names, node_wiz_lineNumber, functions):
    nodes = {}
    edges = []
    node_map = {}
    add_funcs = {}

    node_counter = 0

    for index in range(len(graphs)):
        source = graphs[index].splitlines()
        for i in range(1, len(source)-1):
            cur = source[i]
            if "label = " in source[i]:
                b_id = cur.split(" ")[0]
                tmp = cur.split("[label = \"(")[1][:-4]
                # print(tmp)
                node_type = tmp.split(",")[0]
                if node_type in func_names:
                    add_funcs[node_type] = b_id
                    node_type = "<func>"

                if node_type not in node_type_dict.keys():
                    node_type_dict[node_type] = len(node_type_dict)
                    
                node_type = node_type_dict[node_type]

                node_content = ",".join(tmp.split(",")[1:])
                if 'return' in node_content:
                    node_content = node_content.split(';')[0] + ';'
                

                node_content_ = infercode.encode([node_content])[0].tolist()
                
                line_id = node_wiz_lineNumber[b_id]
                node_map[b_id] = node_counter
                nodes[node_counter] = [node_type, node_content_, line_id, node_content]
                node_counter += 1
            else:
                # if "digraph main {" in graphs[index]:
                edge = cur.split(" -> ")
                # print(edge)
                edges.append([node_map[edge[0][2:]], node_map[edge[1][:-1]]])
                
    for key in add_funcs.keys():
        edges.append([node_map[add_funcs[key]], node_map[functions[key]]])

    input_graph = {'node_type': [i[1][0] for i in nodes.items()],
                    'node_content': [i[1][1] for i in nodes.items()],
                    'node_text': [i[1][3] for i in nodes.items()],
                    'from_idx': [pair[0] for pair in edges], 
                    'to_idx': [pair[1] for pair in edges],
                    'line_id': [i[1][2] for i in nodes.items()],
                }
                
    return input_graph


def extract_graph(dir_path, code_id):
    path = os.path.join(dir_path, code_id)
    graphs = []
    functions = {}
    func_names = []
    node_wiz_lineNumber = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if "dot" in file:
                with open(os.path.join(path,file), 'r') as f:
                    dot_graph = f.read()
                    func_name = dot_graph.split(' ')[1]
                    if "label = " not in dot_graph:
                        continue
                    graphs.append(dot_graph)
                    if func_name != 'main':
                        id_ = extract(dot_graph, func_name)
                        functions[func_name] = id_
                        func_names.append(func_name)
                    else:
                        source = dot_graph.splitlines()
            else:
                with open(os.path.join(path,file), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if 'lineNumber' in item.keys():
                            node_wiz_lineNumber["\"" + str(item['id']) + "\""] = str(item['lineNumber'])
    
    graph = construct_graph(graphs, func_names, node_wiz_lineNumber, functions)
    return graph
        

if __name__=="__main__":
    generate_graph('input/example')
    graph = extract_graph('output','example')

