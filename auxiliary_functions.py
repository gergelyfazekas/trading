

def kwarg_handler(dict, key_string_list):
    if isinstance(key_string_list, list):
        result = [None]*len(key_string_list)
        for i in range(len(result)):
            if isinstance(key_string_list[i], str):
                try:
                    result[i] = dict[key_string_list[i]]
                except KeyError:
                    result[i] = None
            else:
                raise ValueError("key_string_list should be a single str or a list of strs")
        return result
    elif isinstance(key_string_list, str):
        try:
            result = dict[f'{key_text}']
        except KeyError:
            result = None
        return result
    else:
        raise NotImplementedError



a, b, c = kwarg_handler({'A':1, 'B':2}, ['A', 'B', 'C'])



