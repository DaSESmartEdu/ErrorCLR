from tqdm import tqdm
from tree_sitter import Language, Parser
from anytree import AnyNode
from treelib import Tree
import random

def init_parser(language):
    Language.build_library(
        "./build/my-languages.so",
        [
            "./build/tree-sitter-{}".format(language),
        ],
    )
    language = Language("./build/my-languages.so", language)
    lang_parser = Parser()
    lang_parser.set_language(language)
    return lang_parser

parser = init_parser('c')

identifier_type = ["identifier", "field_identifier", "type_identifier", "statement_identifier", \
                    "qualified_identifier", "namespace_identifier"
    ]

string_type = ["string_literal", "concatenated_string", "system_lib_string", "comment", "escape_sequence", "char_literal",
                "raw_string_literal", 
    ]

number_type = ["number_literal"]

opppppps = ["%", "/", ">>", "<<", "&", "--", "++", "-", "|", "^", "+=", "-=", "/=", "%=", "*=",
            ">>=", "&=", "|=", "^=","=",
            "!", "&", "~", "*", "+", "==", "!=", "&&", "||", ">", "<", "<=", ">="]

def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def visualize_tree(any_node):
    tree = Tree()

    def new_tree(node, parent=None):
        if node is None:
            return
        tree.create_node(
            str(node.id)+' '+node.token+('' if not node.terminal else ' '+node.type), node.id, parent=(None if not parent else parent.id)
        )
        for child in node.children:
            #  print(child.token)
            new_tree(child, node)

    new_tree(any_node)

    tree.show()

def create_tree(root, node,node_list,variable_list,token_list, code_lines,parent=None,debug=False, 
                func_list=None, literal_list=None, op_list=None, type_list=None):
    # 根据node id来给节点编号
    node_id = len(node_list)
    node_type = node.type
    children = node.children
    is_terminal = not children
    # print(node_type,is_terminal)

    if node_type == '\n':
        return 

    l_, r_ = node.start_point, node.end_point
    row = l_[0]
    left,right = l_[1], r_[1]
    literal = code_lines[row][left : right]

    if node_type == 'call_expression':
        func_list.append([literal, left, right])
    # elif node_type == 'primitive_type' or node_type == 'sized_type_specifier':
    elif 'type' in node_type:
        type_list.append([literal, left, right])
    elif 'literal' in node_type:
        literal_list.append([literal, left, right])
    elif node_type in opppppps:
        op_list.append([node_type, left, right])
        
    # print(node_type, literal)
    

    # 如果已经是终端节点了
    if is_terminal:
        # print(l_[0], r_[0])
        assert l_[0] == r_[0], 'Terminal node should be at one line.'
        literal = code_lines[row][left : right]

        if node_type in identifier_type:
            # 先将当前的节点加入进去
            current_node = AnyNode(id=node_id,token=literal,type='identifier',terminal=True, parent=parent)
            node_list.append(current_node)
            variable_list.append([literal, left, right])
            if debug:
                print(f'add variable: {literal} type: {node_type}')
        elif is_number(literal) and node_type in number_type:
            # 先将当前的节点加入进去
            current_node = AnyNode(id=node_id,token=literal,type='number',terminal=True,parent=parent)
            node_list.append(current_node)
            # literal_list.append(literal)
            if debug:
                print(f'add number: {literal} type: {node_type}')
        elif node_type in string_type:
            # 先将当前的节点加入进去
            current_node = AnyNode(id=node_id,token=literal,type='string',terminal=True,parent=parent)
            node_list.append(current_node)
            # literal_list.append(literal)
            if debug:
                print(f'add string: {literal} type: {node_type}')
        else:
            current_node = AnyNode(id=node_id,token=literal,type='op',terminal=True,parent=parent)
            node_list.append(current_node)
            # op_list.append(literal)
            if debug:
                print(f'other node: {literal} type: {node_type}')
        token_list.append(node_id)
    else:
        #其他情况，中间节点
        # 空树
        if len(node_list) == 0:
            root.id = node_id
            root.token = node_type
            root.type = node_type
            root.data = node
            current_node = root
            root.terminal = False
        elif node_type in string_type:
            # print('hei')
            current_node = AnyNode(id=node_id,token=literal,type=node_type,terminal=False,parent=parent)
            node_list.append(current_node)
            token_list.append(node_id)
            return
            # print(node.children)
        else:
            current_node = AnyNode(id=node_id,token=node_type,type=node_type,terminal=False,parent=parent)
        node_list.append(current_node)

        for child in children:
            create_tree(root, child, node_list,variable_list,token_list, code_lines,current_node, debug, func_list, literal_list, op_list, type_list)  #,begin=begin
        if debug:
            print(f'un-terminal node: {node_type} type: {node_type}')





def array_related(error_line, correct_line, mask_cnt, target_line=None):
    i = 0
    cnt = 0
    correct_part = re.findall(r'\[(.*?)\]', correct_line)
    while(i < len(error_line)):
        if error_line[i] == '[':
            j = i+1
            while(j < len(error_line) and error_line[j] != ']'):
                j += 1
            if target_line is not None:
                target_line += " <extra_id_" + str(mask_cnt) + "> " + correct_part[cnt]
            error_line = error_line[:i+1] + " <extra_id_" + str(mask_cnt) + "> " + error_line[j:]
            mask_cnt += 1
            cnt += 1
        i += 1

    return error_line, target_line, mask_cnt

def format_related(error_line, correct_line, mask_cnt, target_line=None):
    i = 0
    cnt = 0
    correct_part = re.findall(r'\"(.*?)\"', correct_line)
    # print(correct_part)
    while(i < len(error_line)):
        if error_line[i] == '\"':
            j = len(error_line) - 1
            while(j > i and error_line[j] != '\"'):
                j -= 1
            if target_line is not None:
                target_line += " <extra_id_" + str(mask_cnt) + "> " + correct_part[cnt]
            error_line = error_line[:i+1] + " <extra_id_" + str(mask_cnt) + "> " + error_line[j:]
            mask_cnt += 1
            cnt += 1
            break
        i+=1
    return error_line, target_line, mask_cnt


def fullLine_mask(error_line, correct_line, mask_cnt, target_line=None):
    if target_line is not None:
        target_line += " <extra_id_" + str(mask_cnt) + "> " + correct_line
    error_line = " <extra_id_" + str(mask_cnt) + "> "
    mask_cnt += 1
    return error_line, target_line, mask_cnt

def conditionMissing_mask(error_line, correct_line, mask_cnt, target_line=None):
    if 'for' not in error_line and 'while' not in error_line and \
        'if' not in error_line and 'switch' not in error_line:
        return fullLine_mask(error_line, correct_line, mask_cnt, target_line)

    if 'for' in error_line:
        tmp_error = error_line.split(';')[1]
        tmp_correct = correct_line.split(';')[1]

        error_line = error_line.replace(tmp_error, tmp_error + " <extra_id_" + str(mask_cnt) + "> ")
        if target_line is not None:
            target_line += " <extra_id_" + str(mask_cnt) + "> " + tmp_correct.replace(tmp_error,"")
    else:
        j = len(error_line) - 1
        while(j > 0 and error_line[j] != ')'):
            j -= 1
        error_line = error_line[:j] + " <extra_id_" + str(mask_cnt) + "> " + error_line[j:]
        if target_line is not None: 
            target_line += " <extra_id_" + str(mask_cnt) + "> " + correct_line.replace(error_line[:j],"").replace(error_line[j:],"")

    mask_cnt += 1
    return error_line, target_line, mask_cnt


def initializationMissing_mask(error_line, correct_line, mask_cnt, target_line=None):
    error_candidates = error_line.replace(';','').split(',')
    correct_canditates = correct_line.replace(';','').split(',')
    assert len(error_candidates) == len(correct_canditates)
    error_line = ""

    for i in range(len(error_candidates)):
        error_line += error_candidates[i] + " <extra_id_" + str(mask_cnt) + "> ,"
        if target_line is not None:
            target_line += " <extra_id_" + str(mask_cnt) + "> " + correct_canditates[i].replace(error_candidates[i],"")
        mask_cnt += 1
    error_line = error_line[:-1] + ";"
    
    return error_line, target_line, mask_cnt


def mask_replace(error_line, correct_line, mask_cnt, _list, c_list, target_line=None):
    assert len(_list) == len(c_list)

    cnt = mask_cnt + len(_list) - 1
    target = ""
    i = len(_list)-1
    while i >= 0:
        error_line = error_line[:_list[i][1]] + " <extra_id_" + str(cnt) + "> " + error_line[_list[i][2]:]
        if target_line is not None:
            target = " <extra_id_" + str(cnt) + "> " + correct_line[c_list[i][1]:c_list[i][2]] + target
        mask_cnt += 1
        cnt -= 1
        i -= 1
    if target_line is not None:
        target_line += target
    return error_line, target_line, mask_cnt


def mask_offbyone(error_line, correct_line, mask_cnt, _list, c_list, target_line=None):
    assert len(_list) == len(c_list)

    cnt = mask_cnt + len(_list) - 1
    target = ""
    i = len(_list)-1
    orr = error_line
    flag=False
    while i >= 0:
        error_line = error_line[:_list[i][2]] + " <extra_id_" + str(cnt) + "> " + error_line[_list[i][2]:]
        if target_line is not None:
            if orr[_list[i][2]:] == correct_line[c_list[i][2]:] or flag:
                target = " <extra_id_" + str(cnt) + "> " + target
            else:
                target = " <extra_id_" + str(cnt) + "> " + correct_line[c_list[i][2]:].replace(orr[_list[i][2]:],"") + target
                flag = True
        mask_cnt += 1
        cnt -= 1
        i -= 1
    if target_line is not None:
        target_line += target
    return error_line, target_line, mask_cnt


def complex_mask(error_line, correct_line, type1, type2, mask_cnt, target_line=None):
    code_tree = parser.parse(bytes(error_line, "utf-8"))
    cursor = code_tree.walk()

    variable_list, node_list, token_list, func_list, literal_list, op_list, type_list = [],[],[],[],[],[],[]
    root = AnyNode(id=0,parent=None)

    create_tree(root, cursor.node, node_list, variable_list, token_list, error_line, None, None,
                func_list, literal_list, op_list, type_list)

    ccode_tree = parser.parse(bytes(correct_line, "utf-8"))
    ccursor = ccode_tree.walk()

    cvariable_list, cnode_list, ctoken_list, cfunc_list, cliteral_list, cop_list, ctype_list = [],[],[],[],[],[],[]
    croot = AnyNode(id=0,parent=None)

    create_tree(root, ccursor.node, cnode_list, cvariable_list, ctoken_list, correct_line, None, None,
                cfunc_list, cliteral_list, cop_list, ctype_list)

    if type2 == "WrongOperator":
        return mask_replace(error_line, correct_line, mask_cnt, op_list, cop_list, target_line)
    if type2 == "LiteralMisuse":
        return mask_replace(error_line, correct_line, mask_cnt, literal_list, cliteral_list, target_line)
    if type2 == "VariableMisuse":
        return mask_replace(error_line, correct_line, mask_cnt, variable_list, cvariable_list, target_line)
    if type2 == "DataTypeMisuse" or type2 == "ReturnTypeMisuse" or type2 == "ParameterTypeMisuse":
        return mask_replace(error_line, correct_line, mask_cnt, type_list, ctype_list, target_line)
    if type2 == "FunctionMisuse":
        return mask_replace(error_line, correct_line, mask_cnt, func_list, cfunc_list, target_line)
    if type2 == "OffByOneError":
        return mask_offbyone(error_line, correct_line, mask_cnt, variable_list, cvariable_list, target_line)


def create_mask(error_line, correct_line, type1, type2, mask_cnt, target_line=None):
    if type2 in ["WrongIndex", "WrongArraySize"]:
        return array_related(error_line, correct_line, mask_cnt, target_line)

    if type2 in ["FormatStringMisuse"]:
        return format_related(error_line, correct_line, mask_cnt, target_line)

    if type2 in ["ControlMisuse", "FunctionMissing"] or type2 == "Undefined":
        return fullLine_mask(error_line, correct_line, mask_cnt, target_line)
    
    if type2 in ["ConditionMissing"]:
        return controlMissing_mask(error_line, correct_line, mask_cnt, target_line)

    if type2 in ["InitializationMissing"]:
        return initializationMissing_mask(error_line, correct_line, mask_cnt, target_line)
    
    return complex_mask(error_line, correct_line, type1, type2, mask_cnt, target_line)



def mask_token(line, i, target):
    pass_tokens = ['\'', '\"', ';' , '{', '}', '[', ']', '(', ')', ' ']
    line = line.split(" ")
    valid_ids = [idx for (idx, v) in enumerate(line)]
    random.shuffle(valid_ids)

    for pos in valid_ids:
        if line[pos] not in pass_tokens:
            target += "<extra_id_" + str(i) + ">" + line[pos]
            line[pos] = "<extra_id_" + str(i) + ">"
            i+=1
            break
    
    return " ".join(line), target, i


def mask_subline(line, i, target):
    candidate_sublines = []
    # 1. 分号
    candidate_sublines.extend(re.findall(r'\"(.*?)\"', line))
    # 2. 索引
    candidate_sublines.extend(re.findall(r'\[(.*?)\]', line))
    # 3. 参数部分 / 条件语句
    candidate_sublines.extend(re.findall(r'\((.*?)\)', line))
    if ("for " in line or "for(" in line) and len(candidate_sublines) > 0:
        candidate_sublines.extend(candidate_sublines[-1].split(";"))
    if "if " in line or "if(" in line or "while " in line or "while(" in line:
        if "||" in line and len(candidate_sublines) > 0:
            conditions = candidate_sublines[-1].split("||")
            candidate_sublines.append(conditions[0])
            candidate_sublines.extend(["||" + i for i in conditions[1:]])
        if "||" in line and len(candidate_sublines) > 0:
            conditions = candidate_sublines[-1].split("&&")
            candidate_sublines.append(conditions[0])
            candidate_sublines.extend(["&&" + i for i in conditions[1:]])

    # 4. 表达式部分
    if "=" in line and "!=" not in line and "==" not in line:
        candidate_sublines.extend(line.split("=")[1].strip())
    
    if len(candidate_sublines) == 0:
        return mask_token(line, i, target)

    random.shuffle(candidate_sublines)
    line = line.replace(candidate_sublines[0], "<extra_id_" + str(i) + ">")
    target += "<extra_id_" + str(i) + ">" + candidate_sublines[0]
    i+=1
    return line, target, i
