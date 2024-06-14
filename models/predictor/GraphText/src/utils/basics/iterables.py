def has_intersection(iterable1, iterable2):
    return len(set(iterable1), set(iterable2)) > 0


# * ============================= Itertool Related =============================

def lot_to_tol(list_of_tuple):
    # list of tuple to tuple lists
    # Note: zip(* zipped_file) is an unzip operation
    return list(map(list, zip(*list_of_tuple)))
