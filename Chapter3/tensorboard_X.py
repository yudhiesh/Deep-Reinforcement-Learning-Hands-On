import math
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
    # Here we loop over the angles in radians and then we convert them to radians and calculate our functions values
    for angle in range(-360, 360):
        angel_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angel_rad)
            writer.add_scalar(name, val, angle)
    writer.close()
