import sys
from itertools import islice
from utils import bilinear_upsample_weights
import numpy as np

scale = 2
radius = 2

dsize = radius * scale * 2 + 1

def get_line_number(phrase, file_name):
    with open(file_name) as f:
        for i, line in enumerate(f, 1):
            if phrase in line:
                return i
        return False

def read_weights(file_name, ln, size=0):
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

def format_weights(weights, n):
    return ",".join(['{:.16f}'.format(float(i)) for i in weights.strip(",").split(",")[n:n+4]])

def base_header(file):
    file.write('//!HOOK LUMA\n')
    file.write('//!WHEN OUTPUT.w LUMA.w / {:1.3f} > OUTPUT.h LUMA.h / {:1.3f} > *\n'.format(scale - 0.7, scale - 0.7))

def header1(file, n, d):
    base_header(file)
    file.write('//!DESC feature map {}\n'.format((n//4)%(d//4) + 1))
    file.write('//!BIND LUMA\n')
    file.write('//!SAVE FEATURE{}\n'.format((n//4)%(d//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header2(file, d, n, s):
    base_header(file)
    file.write('//!DESC shrinking {}\n'.format((n//4)%(s//4) + 1))
    for i in range(d//4):
        file.write('//!BIND {}{}\n'.format("FEATURE", i + 1))
    file.write('//!SAVE SHRINKED{}\n'.format((n//4)%(s//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header3(file, r, mi, m, n, s, inp):
    base_header(file)
    file.write('//!DESC mapping {}_{}\n'.format(mi + 1, (n//4)%(s//4) + 1))
    for i in range(s//4):
        file.write('//!BIND {}{}\n'.format(inp if r == 0 and mi == 0 else "MODEL", i+1 + (0 if (r * m + mi) % 2 == 0 else 20)))
    if mi == m-1 and (mi+1)*(r+1) > 1:
        file.write('//!BIND {}{}\n'.format(inp, (n//4)%(s//4) + 1))
    file.write('//!SAVE MODEL{}\n'.format((n//4)%(s//4) + 1 + (20 if (r * m + mi) % 2 == 0 else 0)))
    file.write('//!COMPONENTS 4\n')

def header4(file, s, m, r, n, d):
    base_header(file)
    file.write('//!DESC expanding {}\n'.format((n//4)%(d//4) + 1))
    for i in range(s//4):
        file.write('//!BIND MODEL{}\n'.format(i + 1 + (20 if (r * m) % 2 == 1 else 0)))
    file.write('//!SAVE EXPANDED{}\n'.format((n//4)%(d//4) + 1 + (20 if (r * m) % 2 == 1 else 0)))
    file.write('//!COMPONENTS 4\n')

def header5(file, m, r, n, d, inp):
    base_header(file)
    file.write('//!DESC sub-pixel convolution {}\n'.format((n//4)%(d//4) + 1))
    file.write('//!BIND {}{}\n'.format(inp, (n//4)%(d//4) + 1 + (20 if (r * m) % 2 == 1 else 0)))
    file.write('//!SAVE {}{}\n'.format(inp, (n//4)%(d//4) + 1 + (20 if (r * m) % 2 == 1 else 0)))
    file.write('//!COMPONENTS 4\n')

def header6(file, m, r, d, inp, grl):
    base_header(file)
    file.write('//!WIDTH LUMA.w {} *\n'.format(scale))
    file.write('//!HEIGHT LUMA.h {} *\n'.format(scale))
    file.write('//!DESC aggregation\n')
    if grl:
        file.write('//!BIND HOOKED\n')
    for i in range(d//4):
        file.write('//!BIND {}{}\n'.format(inp, i+1 + (20 if (r * m) % 2 == 1 else 0)))
    file.write('//!OFFSET -{}.0 -{}.0\n'.format(scale//2, scale//2))

def main():
  if len(sys.argv) == 2:
    fname=sys.argv[1]
    d, s, m, r = [int(i) for i in fname[7:fname.index('.')].split("_")]
    if s == 0:
        s = d
        shrinking = False
    else:
        shrinking = True
    dst = fname.replace("_", "-").replace("weights", "FSRCNNX_x{}_r{}_".format(scale, radius)).replace("txt", "glsl")
    with open(dst, 'w') as file:

        # Feature layer
        ln = get_line_number("w1", fname)
        weights = read_weights(fname, ln, (radius*2+1)**2)
        ln = get_line_number("b1", fname)
        biases = read_weights(fname, ln)
        for n in range(0, d, 4):
            header1(file, n, d)
            file.write('vec4 hook()\n')
            file.write('{\n')
            file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
            p = 0
            for l in range(0, len(weights)):
                y, x = p%(radius*2+1)-radius, p//(radius*2+1)-radius
                p += 1
                file.write('res += vec4({}) * float(LUMA_texOff(vec2({},{})));\n'.format(format_weights(weights[l], n), x, y))
            if shrinking:
                ln = get_line_number("alpha1", fname)
                alphas = read_weights(fname, ln)
                file.write('res = max(res, vec4(0.0)) - vec4({}) * max(-res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
            file.write('return res;\n')
            file.write('}\n\n')

        if shrinking:
            # Shrinking layer
            ln = get_line_number("w2", fname)
            weights = read_weights(fname, ln, d)
            ln = get_line_number("b2", fname)
            biases = read_weights(fname, ln)
            for n in range(0, s, 4):
                header2(file, d, n, s)
                file.write('vec4 hook()\n')
                file.write('{\n')
                file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                for l in range(0, d, 4):
                    file.write('res += mat4({},{},{},{}) * FEATURE{}_texOff(vec2(0.0));\n'.format(format_weights(weights[l], n), format_weights(weights[l+1], n), format_weights(weights[l+2], n), format_weights(weights[l+3], n), l//4+1))
                file.write('return res;\n')
                file.write('}\n\n')

        # Mapping layers
        for ri in range(r):
            for mi in range(m):
                ln = get_line_number("w{}".format(mi + 3), fname)
                weights = read_weights(fname, ln, s*9)
                ln = get_line_number("b{}".format(mi + 3), fname)
                biases = read_weights(fname, ln)
                inp = "SHRINKED" if shrinking else "FEATURE"
                for n in range(0, s, 4):
                    header3(file, ri, mi, m, n, s, inp)
                    file.write('vec4 hook()\n')
                    file.write('{\n')
                    file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                    if ri == 0 and mi == 0:
                        file.write('vec4 t;\n')
                    p = 0
                    for l in range(0, len(weights), 4):
                        if l % s == 0:
                            y, x = p%3-1, p//3-1
                            p += 1
                        idx = (l//4)%(s//4)
                        if ri == 0 and mi == 0:
                            ln = get_line_number("alpha3", fname)
                            alphas = read_weights(fname, ln)
                            file.write('t = {}{}_texOff(vec2({},{}));\n'.format(inp, idx + 1, x, y))
                            file.write('res += mat4({},{},{},{}) * (max(t, vec4(0.0)) - vec4({}) * max(-t, vec4(0.0)));\n'.format(
                                        format_weights(weights[l], n), format_weights(weights[l+1], n),
                                        format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                                        format_weights(alphas[0], idx*4)))
                        else:
                            file.write('res += mat4({},{},{},{}) * MODEL{}_texOff(vec2({},{}));\n'.format(
                                        format_weights(weights[l], n), format_weights(weights[l+1], n),
                                        format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                                        idx + 1 + (20 if (ri * m + mi) % 2 == 1 else 0), x, y))
                    ln = get_line_number("alpha{}".format(3 if mi == m - 1 else mi + 4), fname)
                    alphas = read_weights(fname, ln)
                    if mi == m - 1:
                        file.write('res += {}{}_texOff(0);\n'.format(inp, (n//4)%(s//4) + 1))
                        if ri == r - 1:
                            ln = get_line_number("alpha2", fname)
                            alphas = read_weights(fname, ln)
                    file.write('res = max(res, vec4(0.0)) - vec4({}) * max(-res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
                    file.write('return res;\n')
                    file.write('}\n\n')

        if shrinking:
            # Expanding layer
            ln = get_line_number("w{}".format(m + 3), fname)
            weights = read_weights(fname, ln, d)
            ln = get_line_number("b{}".format(m + 3), fname)
            biases = read_weights(fname, ln)
            ln = get_line_number("alpha{}".format(m + 3), fname)
            alphas = read_weights(fname, ln)
            for n in range(0, d, 4):
                header4(file, s, m, r, n, d)
                file.write('vec4 hook()\n')
                file.write('{\n')
                file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                for l in range(0, s, 4):
                    file.write('res += mat4({},{},{},{}) * MODEL{}_texOff(vec2(0.0));\n'.format(format_weights(weights[l], n), format_weights(weights[l+1], n), format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                    l//4 + 1 + (20 if (r * m) % 2 == 1 else 0)))
                file.write('res = max(res, vec4(0.0)) - vec4({}) * max(-res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
                file.write('return res;\n')
                file.write('}\n\n')

        # Sub-pixel convolution
        ln = get_line_number("w{}".format(m + 4), fname)
        weights = read_weights(fname, ln, dsize**2)

        x=list(reversed(range(scale)))
        x=x[-1:]+x[:-1]
        xy = []
        for i in x:
            for j in x:
                xy.append([j, i])

        id = []
        for i in range(0, len(xy)):
            xi, yi = xy[i]
            for y in range(yi, dsize, scale):
                for x in range(xi, dsize, scale):
                    id.append(y + x * dsize)

        weights = list(reversed(weights))
        sort = [weights[id[l]].strip(",") for l in range(0, len(id))]
        inp = "EXPANDED" if shrinking else "MODEL"
        for n in range(0, d, 4):
            header5(file, m, r, n, d, inp)
            file.write('vec4 hook()\n')
            file.write('{\n')
            file.write('vec4 res = vec4(0);\n')
            total = 0
            for i in range(scale):
                for j in range(scale):
                    file.write('res[{}] +=\n'.format(i * scale + j))
                    s2 = radius*2+1 if i == 0 else radius*2
                    for yi, y in enumerate(range(-radius + (0 if i == 0 else 1), radius + 1)):
                        s1 = radius*2+1 if j == 0 else radius*2
                        for xi, x in enumerate(range(-radius + (0 if j == 0 else 1), radius + 1)):
                            l = yi * s1 + xi
                            file.write('dot(vec4({}), {}{}_texOff(vec2({},{}))){}\n'.format(format_weights(sort[l+total], n), inp,
                                        (n//4)%(d//4) + 1 + (20 if (r * m) % 2 == 1 else 0), x, y, ';' if l == s1 * s2 - 1 else '+'))
                    total = total + l + 1
            file.write('return res;\n')
            file.write('}\n\n')

        # Aggregation
        ln = get_line_number("b{}".format(m + 4), fname)
        biases = read_weights(fname, ln)
        grl = get_line_number("b{}".format(m + 5), fname)
        header6(file, m, r, d, inp, grl)
        file.write('vec4 hook()\n')
        file.write('{\n')
        file.write('float res = {};\n'.format(float(biases[0]) + (float(read_weights(fname, grl)[0]) if grl else 0.0)))
        v = 1 + (20 if (r * m) % 2 == 1 else 0)
        file.write('vec2 fcoord = fract({}{}_pos * {}{}_size);\n'.format(inp, v, inp, v))
        file.write('vec2 base = {}{}_pos + (vec2(0.5) - fcoord) * {}{}_pt;\n'.format(inp, v, inp, v))
        file.write('ivec2 index = ivec2(fcoord * vec2({}));\n'.format(scale))
        file.write('res += ({}{}_tex(base)'.format(inp, v))
        for i in range(d//4-1):
            file.write('+{}{}_tex(base)'.format(inp, i + 1 + v))
        file.write(')[index.y * {} + index.x];\n'.format(scale))

        if grl:
            weights = bilinear_upsample_weights(scale, 1).ravel(order='F')
            x=list(reversed(range(scale)))
            xy = []
            for i in x:
                for j in x:
                    xy.append([j, i])
            id = []
            for i in range(len(xy)):
                xi, yi = xy[i]
                for y in range(yi, scale * 2, scale):
                    for x in range(xi, scale * 2, scale):
                        id.append(y + x * (scale * 2))
            sort = [weights[id[l]] for l in range(0, len(id))]
            file.write('vec4 img = vec4(0);\n')
            for i in range(scale):
                for j in range(scale):
                    idx = i * scale + j
                    file.write('img[{}] =\n'.format(idx))
                    file.write('{} * float(HOOKED_tex(base + HOOKED_pt * vec2(0,0)))+\n'.format(sort[idx * 4]))
                    file.write('{} * float(HOOKED_tex(base + HOOKED_pt * vec2(1,0)))+\n'.format(sort[idx * 4 + 1]))
                    file.write('{} * float(HOOKED_tex(base + HOOKED_pt * vec2(0,1)))+\n'.format(sort[idx * 4 + 2]))
                    file.write('{} * float(HOOKED_tex(base + HOOKED_pt * vec2(1,1)));\n'.format(sort[idx * 4 + 3]))
            file.write('res += img[index.y * {} + index.x];\n'.format(scale))
        file.write('return vec4(res, 0, 0, 1);\n')
        file.write('}\n')

  else:
    print("Missing argument: You must specify a file name")
    return

if __name__ == '__main__':
  main()
