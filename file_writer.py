



def write_to_lg(file, symbol_list, relations, labels, lg_dir):

    # Create object strings
    object_strings = "# IUD, " + str(file).strip(".inkml") + "\n"
    object_strings += "# Objects(" + str(len(symbol_list)) + ") : \n"
    for symbol in symbol_list:
        string = "O, " + symbol.symbol + "_" + str(symbol.sym_ct) + ", " + symbol.symbol + ", " + "1.0, "
        strokes = ""
        for s in symbol.stroke_id:
            strokes += str(s) + ", "

        strokes = strokes.strip(", ")

        string += strokes

        object_strings += string + "\n"

    # print(object_strings)

    # print("Relations : ", relations)
    # Create relation strings
    relation_string = "# Relations from SRT: \n"
    for key in relations:
        for k in relations[key]:
            line = "R, " + symbol_list[key].symbol + "_" + str(symbol_list[key].sym_ct) + ", " + symbol_list[
                k].symbol + "_" + str(symbol_list[k].sym_ct) + ", " + labels[key][k] + ", " + "1.0\n"

            relation_string += line


    # Write to file
    lg_file_name = file.strip().split('.inkml')[0] + ".lg"
    lg_file = open(lg_dir + "/" + lg_file_name, 'w+')
    lg_file.write(object_strings)
    lg_file.write("\n")
    lg_file.write(relation_string)
    lg_file.close()
