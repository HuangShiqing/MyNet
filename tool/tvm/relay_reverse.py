import re
import argparse


class layer:
    def __init__(self, name, type, bottoms, tops, params):
        self.name = name
        self.type = type
        self.bottoms = bottoms
        self.tops = tops
        self.params = params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        help="The path of the relay text txt.",
    )
    args = parser.parse_args()
    relay_text_path = args.input_path
    caffe_prototxt_path = relay_text_path.replace(".txt", ".prototxt")
    relay_python_path = relay_text_path.replace(".txt", ".py")

    # TODO:
    # 1. param name include . or {
    #    %0 = nn.conv2d(%input_feature, meta[relay.Constant][0] /* ty=Tensor[(64, 74, 1, 1), float32] */, out_max=None, relay.attrs.Conv2DAttrs{padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]}) /* span = Conv_0 */ /* ty=Tensor[(1, 64, 1152, 1152), float32] */;
    with open(relay_text_path, 'r') as f:
        layer_list = list()
        replace_map = dict()
        net_metas = list()
        net_meta_shapes = dict()
        net_meta_dtypes = dict()
        net_inputs = None
        net_input_shapes = None
        output_count = 0
        scale_count = 0
        for line in f:
            # net_inputs = list()
            # line = "  %114 = subtract(1f /* ty=float32 */, %90) /* ty=Tensor[(1, 12, 50), float32] */;"
            # step 0. get type
            # -1: unkonw
            # 0: "def @main(%INPUT__0..."
            # 1: "%0 = (%INPUT__0, %INPUT__1)"
            # 2: "%7 = %3.0" or "%99 = (%97, %98);"
            # 3: "%9 = nn.scale("
            # 4: "nn.fully_connected(%144..."
            # 5: "}"
            type = -1
            if "def" in line:
                type = 0
            elif "}" in line and len(line) == 2:# len("}\n")==2:
                type = 5
            else:
                index = line.find('(')
                if index == -1:
                    type = 2
                else:
                    index2 = line.find('=')
                    if index-index2 == 2:
                        type = 1
                    else:
                        if line[2] != '%':
                            type = 4
                        else:
                            type = 3

            # step 1. process
            if type == -1:
                raise
            elif type == 0:  # 0: "def @main(%INPUT__0..."
                bottoms = [i.split(':')[0] for i in line.split("%")[1:]]
                shapes = [list(i.split(')')[0].split(', '))
                          for i in line.split("Tensor[(")[1:]]

                net_inputs = bottoms
                net_input_shapes = shapes
            # 1: "  %0 = (%INPUT__0, %INPUT__1)"
            # 2: "  %7 = %3.0;\n" or "%99 = (%97, %98);"
            elif type == 1 or type == 2:
                bottoms = [i.split(',')[0].split(';')[0].split(')')[0]
                           for i in line.split("%")[2:]]
                bottoms = ["call_"+bottom if bottom not in net_inputs else bottom for bottom in bottoms ]
                top = "call_"+line.split("%")[1].split(' =')[0]

                for bottom in bottoms:
                    if '.' in bottom:
                        a = int(bottom.split('.')[0].strip("call_"))
                        i = int(bottom.split('.')[1])
                        if i == 0:
                            layer_list[a].tops[0] += '.0'
                        else:
                            while i > len(layer_list[a].tops)-1:
                                layer_list[a].tops.append(
                                    "call_"+str(a)+'.'+str(len(layer_list[a].tops)))
                    else:
                        continue
                if "meta[relay.Constant]" in line:
                    constant_bottoms = ["Constant_"+i.split(']')[0] for i in line.split("meta[relay.Constant][")[1:]]
                    constant_shape = [i.split("ty=Tensor[")[1].split("), ")[0]+')' for i in line.split("meta[relay.Constant]")[1:]]
                    constant_dtype = [i.split("ty=Tensor[")[1].split("), ")[1].split(']')[0] for i in line.split("meta[relay.Constant]")[1:]]
                    net_metas += constant_bottoms
                    for i, const in enumerate(constant_bottoms):
                        net_meta_shapes[const] = constant_shape[i]
                        net_meta_dtypes[const] = constant_dtype[i]
                    bottoms += constant_bottoms
                # reorder bottoms
                def key_fun(bottom):
                    if "Constant" in bottom:
                        constant_bottom = "meta[relay.Constant][{}]".format(bottom.split('_')[1])
                        raw_index = line.find(constant_bottom)
                    # TODO: shall will this type looks like "%0 = (%INPUT__0, 1 /* ty=int64 */)"
                    else:
                        raw_index = line.find("%"+bottom.strip("call_"))
                    return raw_index
                bottoms.sort(key=key_fun)

                replace_map[top] = bottoms
                layer_list.append(None)
            elif type == 3 or type == 4:
                # process link
                # 3: "  %12 = nn.fused_qkv_attention(%10, %11, ..."
                if type == 3:
                    name = line.split(" = ")[0].strip(" ")
                    type = line.split(" = ")[1].split("(")[0]
                    bottoms = [i.split(',')[0].split(')')[0]
                               for i in line.split("%")[2:]]
                    bottoms = ["call_"+bottom if bottom not in net_inputs else bottom for bottom in bottoms ]
                    tops = ["call_"+line.split("%")[1].split(' =')[0]]
                elif type == 4:  # 4: "  nn.fully_connected(%144..."
                    name = ""
                    type = line.split("(%")[0].strip()
                    bottoms = [i.split(',')[0].split(')')[0]
                               for i in line.split("%")[1:]]
                    bottoms = ["call_"+bottom if bottom not in net_inputs else bottom for bottom in bottoms ]
                    tops = ["call_"+"output"+str(output_count)]
                    output_count += 1

                # %134 = take(%72, 0 /* ty=int64 */, axis=1)
                # %114 = subtract(1f /* ty=float32 */, %90)
                bottoms += [i.split(' ')[0].strip('f') for i in ''.join(line.split("(")[1:]).split(", ") if ' ' in i and i.split(' ')[0].strip('f').isdigit() == True ]

                if "meta[relay.Constant]" in line:
                    constant_bottoms = ["Constant_"+i.split(']')[0] for i in line.split("meta[relay.Constant][")[1:]]
                    constant_shape = [i.split("ty=Tensor[")[1].split("), ")[0]+')' for i in line.split("meta[relay.Constant]")[1:]]
                    constant_dtype = [i.split("ty=Tensor[")[1].split("), ")[1].split(']')[0] for i in line.split("meta[relay.Constant]")[1:]]
                    net_metas += constant_bottoms
                    for i, const in enumerate(constant_bottoms):
                        net_meta_shapes[const] = constant_shape[i]
                        net_meta_dtypes[const] = constant_dtype[i]
                    bottoms += constant_bottoms

                # reorder
                def key_fun(bottom):
                    raw_index = -1
                    if "Constant" in bottom:
                        constant_bottom = "meta[relay.Constant][{}]".format(bottom.split('_')[1])
                        raw_index = line.find(constant_bottom)
                    elif "call_" not in bottom:
                        if line.find('('+bottom) != -1:
                            raw_index = line.find('('+bottom)
                        elif line.find(', '+bottom) != -1:
                            raw_index = line.find(', '+bottom)
                    else:
                        raw_index = line.find("%"+bottom.strip("call_"))
                    return raw_index
                bottoms.sort(key=key_fun)

                new_bottoms = list()
                for bottom in bottoms:
                    if replace_map.get(bottom) != None:
                        new_bottoms.append(replace_map[bottom])
                    else:
                        new_bottoms.append(bottom)

                # process param
                params = dict()
                s = line.split('=')
                for i in range(1, len(s)):
                    if s[i-1][-1] == ' ' and s[i][0] == ' ':  # %13 = nn.fully_connected
                        continue
                    # /* ty=Tensor[(1, 1, 50), float32] */;
                    if "/*" in s[i-1] and "*/" in s[i]:
                        continue
                    param = s[i-1].split(", ")[-1]
                    value = s[i].split(", ")[0].split(")")[0]
                    # out_shape=[1, 64, 768],  axes=[0, 2, 3, 1]) /* ty=T
                    if s[i].split(", ")[0][0] == '[':
                        value = s[i].split("]")[0]+']'
                    else:
                        value = s[i].split(", ")[0].split(")")[0]
                    # epsilon: 1e-08f     bias: -10000f
                    if value[-1] == 'f':
                        value = value.replace('f', '')
                    params[param] = value

                # TODO: process shape
                l = layer(name=name, type=type, bottoms=new_bottoms, tops=tops, params=params)
                layer_list.append(l)
            elif type == 5:
                pass

    type_transform_map = {
        "dyn.reshape":"reshape",
        "dyn.strided_slice":"strided_slice"
    }
    type_param_ignore_map = {
        "dyn.reshape":["newshape"],
        "dyn.strided_slice":["begin", "end", "strides"],
    }
    type_param_transform_map = {
        "nn.gelu":{"old":"format", "new":"gelu_format"}
    }
    param_ignore_list = ["out_dtype"]
    # create relay.prototxt
    with open(caffe_prototxt_path, 'w') as caffe_prototxt:
        for index, net_input in enumerate(net_inputs):
            caffe_prototxt.write("input: \"{}\"\r".format(net_input))
            caffe_prototxt.write("input_shape {\r")
            for dim in net_input_shapes[index]:
                caffe_prototxt.write("  dim: {}\r".format(
                    dim if dim != '?' else '-1'))
            caffe_prototxt.write("}\r")

        for l in layer_list:
            if l == None:
                continue
            caffe_prototxt.write("layer {\r")
            caffe_prototxt.write("  name: \"{}\"\r".format(l.name))
            caffe_prototxt.write("  type: \"{}\"\r".format(l.type))
            # TODO: why all bottoms' index are 0 in netron
            for bottom in l.bottoms:
                for b in bottom if isinstance(bottom, list) else [bottom]:
                    caffe_prototxt.write("  bottom: \"{}\"\r".format(b))
            for top in l.tops:
                caffe_prototxt.write("  top: \"{}\"\r".format(top))

            # this param only have one field
            caffe_prototxt.write("  hdf5_output_param {\r")
            for key in l.params:
                # ignore some param
                if l.type in type_param_ignore_map:
                    if key in type_param_ignore_map[l.type]:
                        continue
                if key in param_ignore_list:
                    continue
                caffe_prototxt.write(
                    "    {}: {}\r".format(key, l.params[key]))
            caffe_prototxt.write("  }\r")

            caffe_prototxt.write("}\r")

    # create relay_python.py
    with open(relay_python_path, 'w') as relay_python:
        relay_python.write("from tvm import relay, IRModule\r")
        relay_python.write("import numpy as np\r\r")
        relay_python.write("def Module():\r")
        for index, net_input in enumerate(net_inputs):
            relay_python.write(
                "    {} = relay.var(\"{}\", shape=(".format(net_input, net_input))
            for dim in net_input_shapes[index]:
                relay_python.write("{}, ".format("relay.Any()" if dim == '?' else dim))
            relay_python.write("), dtype=\"float32\")\r")  # TODO:type

        for l in layer_list:
            if l == None:
                continue

            for i, bottom in enumerate(l.bottoms):
                for b in bottom if isinstance(bottom, list) else [bottom]:
                    if "Constant" in b:
                        if "int" in net_meta_dtypes[b]:
                            relay_python.write("    {} = relay.const(np.random.randint(0, 10, ({})), dtype=\"{}\")\r".format(b, net_meta_shapes[b].strip('(').strip(')'), net_meta_dtypes[b]))
                        else:
                            relay_python.write("    {} = relay.const(np.random.rand({}), dtype=\"{}\")\r".format(b, net_meta_shapes[b].strip('(').strip(')'), net_meta_dtypes[b]))
                    # if b.isdigit() == True:
                    #     relay_python.write("{} = relay.const(np.array({}, dtype=\"float32\"))\r".format("scale_"+b, b))# TODO:type
            for i, top in enumerate(l.tops):
                relay_python.write("    {}".format(top.replace('.', '_')))
                if i != len(l.tops) - 1:
                    relay_python.write(", ")
            relay_python.write(" = relay.{}(".format(type_transform_map[l.type] if l.type in type_transform_map else l.type))

            for i, bottom in enumerate(l.bottoms):
                if isinstance(bottom, list):
                    relay_python.write("relay.Tuple([")
                for j, b in enumerate(bottom if isinstance(bottom, list) else [bottom]):
                    if b.isdigit() == True:
                        relay_python.write("relay.const(np.array({}, dtype=\"int64\"))".format(b))# TODO:type
                        # relay_python.write("{}".format("scale_"+b))
                    else:
                        relay_python.write("{}".format(b.replace('.', '_')))
                    if j != len(bottom if isinstance(bottom, list) else [bottom]) - 1:
                        relay_python.write(", ")
                if isinstance(bottom, list):
                    relay_python.write("])")
                if i != len(l.bottoms) - 1:
                    relay_python.write(", ")
            relay_python.write(", ")

            for i, key in enumerate(l.params):
                # ignore some param
                if l.type in type_param_ignore_map:
                    if key in type_param_ignore_map[l.type]:
                        continue
                if key in param_ignore_list:
                    continue
                relay_python.write("{}={}".format(key, l.params[key]))
                if i != len(l.params) - 1:
                    relay_python.write(", ")
            relay_python.write(")\n")

        relay_python.write("    return ")
        for i in range(output_count):
            relay_python.write("call_output{}".format(i))
            if i != output_count - 1:
                relay_python.write(", ")

if __name__ == "__main__":
    main()
