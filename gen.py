import sys
from itertools import islice

radius = 2

def get_line_number(phrase, file_name):
    with open(file_name) as f:
        for i, line in enumerate(f, 1):
            if phrase in line:
                return i

def read_weights(file_name, ln, size):
    content = []
    off = 1 if size == 0 else 0
    with open(file_name) as f:
        for line in islice(f, ln - off, ln + size):
            if line.find('[') != -1:
                line = line[line.index('[') + 1:]
            if line.find(']') != -1:
                line = line[:line.rindex(']')]
            content.append(line)

    return [x.strip() for x in content]

def header1(file, n, d):
    file.write('//!HOOK LUMA\n')
    file.write('//!DESC feature map {}\n'.format((n//4)%(d//4) + 1))
    file.write('//!BIND LUMA\n')
    file.write('//!SAVE MODEL{}\n'.format((n//4)%(d//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header2(file, w, n, s):
    file.write('//!HOOK LUMA\n')
    file.write('//!DESC mapping {}_{}\n'.format(w+1, (n//4)%(s//4) + 1))
    for i in range(s//4):
        if (w+1) % 2 == 1:
            file.write('//!BIND MODEL{}\n'.format(i+1))
        else:
            file.write('//!BIND MODEL{}{}\n'.format(2, i+1))
    if (w+1) % 2 == 1:
        file.write('//!SAVE MODEL{}{}\n'.format(2, (n//4)%(s//4) + 1))
    else:
        file.write('//!SAVE MODEL{}\n'.format((n//4)%(s//4) + 1))
    file.write('//!COMPONENTS 4\n')

def main():
  if len(sys.argv) == 2:
    fname=sys.argv[1]
    d, s, m = [int(i) for i in fname[7:fname.index('.')].split("_")]
    if s == 0:
        s = d
    dst = fname.replace("weights", "FSRCNN_").replace("txt", "glsl")
    with open(dst, 'w') as file:

        # Feature layer
        ln = get_line_number("w1", fname)
        weights = read_weights(fname, ln, (radius*2+1)**2)
        ln = get_line_number("b1", fname)
        biases = read_weights(fname, ln, 0)
        ln = get_line_number("alpha1", fname)
        alphas = read_weights(fname, ln, 0)
        for n in range(0, d, 4):
            header1(file, n, d)
            file.write('vec4 hook()\n')
            file.write('{\n')
            file.write('vec4 res = vec4({});\n'.format(",".join(biases[0].strip(",").split(",")[n:n+4])))
            p = 0
            for l in range(0, len(weights)):
                y, x = p%(radius*2+1)-2, p//(radius*2+1)-2
                p += 1
                file.write('res += vec4({}) * float(LUMA_texOff(vec2({},{})));\n'.format(",".join(weights[l].strip(",").split(",")[n:n+4]), x, y))
            file.write('res = mix(res, vec4({}) * res, lessThan(res, vec4(0.0)));\n'.format(",".join(alphas[0].strip(",").split(",")[n:n+4])))
            file.write('return res;\n')
            file.write('}\n\n')

        # Mapping layers
        for w in range(m):
            ln = get_line_number("w{}".format(w + 3), fname)
            weights = read_weights(fname, ln, s*9)
            ln = get_line_number("b{}".format(w + 3), fname)
            biases = read_weights(fname, ln, 0)
            ln = get_line_number("alpha{}".format(w + 3), fname)
            alphas = read_weights(fname, ln, 0)
            for n in range(0, s, 4):
                header2(file, w, n, s)
                file.write('vec4 hook()\n')
                file.write('{\n')
                file.write('vec4 res = vec4({});\n'.format(",".join(biases[0].strip(",").split(",")[n:n+4])))
                p = 0
                for l in range(0, len(weights), 4):
                    if l % s == 0:
                        y, x = p%3-1, p//3-1
                        p += 1
                    file.write('res += mat4({},{},{},{}) * vec4(MODEL{}_texOff(vec2({},{})));\n'.format(",".join(weights[l].strip(",").split(",")[n:n+4]), ",".join(weights[l+1].strip(",").split(",")[n:n+4]), ",".join(weights[l+2].strip(",").split(",")[n:n+4]), ",".join(weights[l+3].strip(",").split(",")[n:n+4]), (l//4)%(s//4) + 1 + (20 if (w+1) % 2 == 0 else 0), x, y))
                file.write('res = mix(res, vec4({}) * res, lessThan(res, vec4(0.0)));\n'.format(",".join(alphas[0].strip(",").split(",")[n:n+4])))
                file.write('return res;\n')
                file.write('}\n\n')

  else:
    print("Missing argument: You must specify a file name")
    return

if __name__ == '__main__':
  main()
