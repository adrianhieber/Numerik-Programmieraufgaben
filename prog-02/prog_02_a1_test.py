import numpy as np
import poisson_skeleton as prog


def test():
    obj = prog.Simple_DGL()

    interval = [0, 1]

    # boundary
    act = obj.boundary()
    exp = [1, 1]
    if act == exp:
        print(f"Successful: Boundary Test with I={interval}")
    else:
        print(f"Failed: Boundary Test with I={interval}: act={act} exp={exp}")

    # call with 0.15
    act = obj(0.15)
    exp = 1.0353812
    if np.isclose(act, exp):
        print(f"Successful: call Test with I={interval}")
    else:
        print(f"Failed: call Test with I={interval}: act={act} exp={exp}")

    # rhs with a
    act = float(obj.rhs(obj.a))
    exp = 4
    if np.isclose(act, exp):
        print(f"Successful: rhs(a) Test with I={interval}")
    else:
        print(f"Failed: rhs(a) Test with I={interval}: act={act} exp={exp}")

    # rhs with b
    act = float(obj.rhs(obj.b))
    exp = -2
    if np.isclose(act, exp):
        print(f"Successful: rhs(b) Test with I={interval}")
    else:
        print(f"Failed: rhs(b) Test with I={interval}: act={act} exp={exp}")

    # rhs test with val
    val = 0.25
    act = float(obj.rhs(val))
    exp = 0.25
    if np.isclose(act, exp):
        print(f"Successful: rhs({val}) Test with I={interval}")
    else:
        print(f"Failed: rhs({val}) Test with I={interval}: act={act} exp={exp}")

    # rhs test with val
    val = 0.1
    act = float(obj.rhs(val))
    exp = 2.32
    if np.isclose(act, exp):
        print(f"Successful: rhs({val}) Test with I={interval}")
    else:
        print(f"Failed: rhs({val}) Test with I={interval}: act={act} exp={exp}")

    # rhs test with val
    val = 0.15
    act = float(obj.rhs(val))
    exp = 1.57
    if np.isclose(act, exp):
        print(f"Successful: rhs({val}) Test with I={interval}")
    else:
        print(f"Failed: rhs({val}) Test with I={interval}: act={act} exp={exp}")
