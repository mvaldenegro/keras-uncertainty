import numpy as np

# To make batches from array or iterable, reference https://stackoverflow.com/a/8290508/349130
def make_batches(iter_x, iter_y, batch_size=32):
    l = len(iter_x)
 
    for ndx in range(0, l, batch_size):
        x = iter_x[ndx:min(ndx + batch_size, l)]
        y = iter_y[ndx:min(ndx + batch_size, l)]
        
        yield x, y

# To make batches from array or iterable, reference https://stackoverflow.com/a/8290508/349130
def predict_batches(predict_fn, iterable, batch_size, progress_bar=False):
    l = len(iterable)    

    if progress_bar:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x

    if batch_size > 1:
        output = [None] * l
        for ndx in tqdm(range(0, l, batch_size)):
            inp = iterable[ndx:min(ndx + batch_size, l)]
            output[ndx:min(ndx + batch_size, l)] = predict_fn(inp)

        return output
    else:
        output = []
        for ndx in tqdm(range(0, l, batch_size)):
            inp = iterable[ndx:min(ndx + batch_size, l)]
            value = predict_fn(inp)
            output.append(value)

        return output
