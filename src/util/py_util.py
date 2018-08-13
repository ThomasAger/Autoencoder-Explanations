from util import io

def parameter_list_to_dict_str(parameter_list_string):#
    dict_str = ["param_dict = {"]
    for i in range(len(parameter_list_string)):
        str = ""
        if parameter_list_string[i][:1] == "#":
            continue
        else:
            split = parameter_list_string[i].split()
            if len(split) == 0:
                continue
            str += "\t'" + split[0] + "': " + split[0] + ","
        dict_str.append(str)
    dict_str.append("}")
    return dict_str

if __name__ == '__main__':
    parameter_list_string = io.import1dArray("../../data/parameter_list_string.txt")
    parameter_dict = parameter_list_to_dict_str(parameter_list_string)
    io.write1dArray(parameter_dict, "../../data/parameter_dict.txt")