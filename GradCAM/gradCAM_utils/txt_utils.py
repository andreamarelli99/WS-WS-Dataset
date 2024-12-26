def read_txt(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_txt(path, data_list):
    with open(path, 'w') as f:
        for data in data_list:
            f.write(data + '\n')

def add_txt(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')

def log_print(message, path):
    """This function shows message and saves message.
    
    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.
        
        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
    print(message)
    add_txt(path, message)