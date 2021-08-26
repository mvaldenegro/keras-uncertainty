import numpy as np

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
            output[ndx:min(ndx + batch_size, l)] = predict_fn(inp)[0]

        return output
    else:
        output = []
        for ndx in tqdm(range(0, l, batch_size)):
            inp = iterable[ndx:min(ndx + batch_size, l)]
            value = predict_fn(inp)[0]
            output.append(value)

        return output
