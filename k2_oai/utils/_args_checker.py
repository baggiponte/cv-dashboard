def is_positive_odd_integer(x: int) -> None:
    """Checks the argument is positive odd integer; else raises ValueError.

    Parameters
    ----------
    x : int
        The argument to check.

    Returns
    -------
    None
    """
    if isinstance(x, int) and x > 0 and x % 2 != 0:
        return None
    else:
        raise ValueError(f"{x} must be an odd, positive integer.")
