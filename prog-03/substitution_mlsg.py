import numpy as np


def check_input(A, b, criterions=[], tol=1e-12):
    # Überprüfe ob A zweidimensional
    if ("2-dim" in criterions) and (np.ndim(A) != 2):
        raise (ValueError("Die Matrix ist nicht zwei-dimensional!"))

    # Überprüfe ob A quadratisch
    if ("quadratic" in criterions) and (A.shape[0] != A.shape[1]):
        raise (ValueError("Die Matrix ist nicht quadratisch."))

    # Überprüfe ob A obere Dreiecksmatirx
    if ("triangular" in criterions) and np.any(np.tril(A, -1)):
        raise (ValueError("Die Matrix ist keine obere Diagonalmatrix."))

    # Überprüfe ob A nullen auf der Diagonale hat
    if ("zero-diag" in criterions) and (np.min(np.abs(np.diag(A))) < tol):
        raise (ValueError("Die Matrix hat Nullen auf der Diagonale."))

    # Überprüfe, dass b im richtigen Format ist
    if np.ndim(b) != 1:
        raise (ValueError("Der Vektor soll Shape (n,) haben!"))

    # Überprüfe, dass und b kompatibel sind.
    if b.shape[0] != A.shape[1]:
        raise (ValueError("Matrix und Vektor sind nicht kompatibel!"))


def backwardSubstitution(R, b):
    """backwardSubstitution
    ========
    Implementiert das Rückwärtseinsetzten zum Lösen eines LGS.
    ----------
    Arguments:
        R (np.array): Untere Dreiecksmatrix.
        b (np.array): Rechte Seite.

    Returns:
        x (np.array): Lösungsvektor.
    """
    n = np.shape(R)[0]
    # Überprüfe notwendige Eigenschaften der Variablen R und b
    check_input(
        R, b, criterions=["2-dim", "quadratic", "zero-diag", "upper-triangular"]
    )
    # initialisiere ein Vektor, welcher später zurückgegeben wird.
    x = np.zeros(n)
    # berechne letzten Eintrag von x
    x[-1] = b[-1] / R[-1, -1]
    # durchlaufe die übergebenen Gleichungen in umgekehrter Reihenfolge
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - R[i, i:] @ x[i:]) / R[i, i]
    # gib Lösungsvektor x zurück
    return x


#%%
def forwardSubstitution(R, b):
    """forwardSubstitution
    ========
    Implementiert das Vorwärtseinsetzten zum Lösen eines LGS.
    ----------
    Arguments:
        R (np.array): Untere Dreiecksmatrix.
        b (np.array): Rechte Seite.

    Returns:
        x (np.array): Lösungsvektor.
    """
    n = np.shape(R)[0]
    # Überprüfe notwendige Eigenschaften der Variablen R und b
    check_input(
        R, b, criterions=["2-dim", "quadratic", "zero-diag", "lower-triangular"]
    )
    # initialisiere ein Vektor, welcher später zurückgegeben wird.
    x = np.zeros(n)
    # berechne letzten Eintrag von x
    x[0] = b[0] / R[0, 0]
    # durchlaufe die übergebenen Gleichungen in umgekehrter Reihenfolge
    for i in range(1, n):
        x[i] = (b[i] - R[i, :i] @ x[:i]) / R[i, i]
    # gib Lösungsvektor x zurück
    return x
