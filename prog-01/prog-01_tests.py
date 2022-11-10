def __printReturns():
    print("\nPrint returns:")
    A = np.array([[1, 3, 2, 7], [1, 2, 3, 2], [5, 2, 3, 6], [5, 0, 3, 7]])
    B = np.array([[0, 3, 0, 7], [0, 0, 3, 2], [5, 2, 0, 6], [5, 0, 0, 7]])
    b = np.array([1, 2, 5, 9])

    actX, P, L, R = solveLES(A.copy(), b.copy())
    print(f"\nInput:\nA=\n{A}\nb={b}")
    print(f"\nOutput:\nx={actX}\nP=\n{P}\nL=\n{L}\nR=\n{R}")
    print()
    actX, P, L, R = solveLES(B.copy(), b.copy())
    print(f"\nInput:\nB=\n{B}\nb={b}")
    print(f"\nOutput:\nx={actX}\nP=\n{P}\nL=\n{L}\nR=\n{R}")


def __testSolveLESforX():
    print("Test solveLES():")
    A = np.array([[1, 3, 2, 7], [1, 2, 3, 2], [5, 2, 3, 6], [5, 0, 3, 7]])
    B = np.array([[0, 3, 0, 7], [0, 0, 3, 2], [5, 2, 0, 6], [5, 0, 0, 7]])
    b = np.array([1, 2, 5, 9])

    # without pivotSearch
    expX = np.linalg.solve(A.copy(), b.copy())
    actX, P, L, R = solveLES(A.copy(), b.copy())
    if not np.allclose(actX, expX):
        print(f"FAILED: Calculation wrong: \nexp:{expX} \nact:{actX}")
        return
    if not np.allclose(P @ A, L @ R):
        print(f"FAILED: Calculation wrong with PA=LR: \nPA=\n{P@A} \nLR=\n{L@R}")
        return
    print("\tSuccessful: Calculation without pivotSearch test")

    # with pivotSearch
    expX = np.linalg.solve(B.copy(), b.copy())
    actX, P, L, R = solveLES(B.copy(), b.copy())
    if not np.allclose(actX, expX):
        print(f"FAILED: Calculation wrong: \nexp:{expX} \nact:{actX}")
        return
    if not np.allclose(P @ B, L @ R):
        print(f"FAILED: Calculation wrong with PA=LR: \nPA=\n{P@A} \nLR=\n{L@R}")
        return
    print("\tSuccessful: Calculation with pivotSearch test")

    print("Complete: solveLES() test \n")


def __testBackSubstitutionForX():
    print("Test backSubstitution():")
    A = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 1]])  # not invertible
    B = np.array([[1, 0, 1], [0, 1, 0]])  # not square
    C = np.array([[1, 3, 2], [1, 2, 3], [5, 2, 3]])  # not upper triangular matrix
    D = np.array([[1, 3, 2], [0, 2, 3], [0, 0, 3]])  # should work
    y = np.array([1, 2, 5])
    ySmall = np.array([0, 1])

    try:  # test square
        backSubstitution(B, y)
        print("FAILED: not square test")
        return
    except Exception:
        print("\tSuccessful: not square test")

    try:  # dimensions test
        backSubstitution(D, ySmall)
        print("\tFAILED: dimensions test")
        return
    except Exception:
        print("\tSuccessful: dimensions test")

    try:  # invertible test
        backSubstitution(A, y)
        print("FAILED: not invertible test")
        return
    except Exception:
        print("\tSuccessful: not invertible test")

    try:  # upper triangular matrix test
        backSubstitution(C, y)
        print("FAILED: not upper triangular matrix test")
        return
    except Exception:
        print("\tSuccessful: not upper triangular matrix test")

    # calculation test
    expX = np.linalg.solve(D.copy(), y)
    actX = backSubstitution(D.copy(), y)
    if not np.array_equal(actX, expX):
        print(f"FAILED: Calculation wrong: \nexp:{expX} \nact:{actX}")
        return
    print("\tSuccessful: Calculation test")

    print("Complete: backSubstitution() test \n")

def test():
    __testBackSubstitutionForX()
    __testSolveLESforX()
    __printReturns()
