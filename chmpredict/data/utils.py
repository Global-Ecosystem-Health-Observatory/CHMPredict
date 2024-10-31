import os


def create_file_pairs(rgb_dir, chm_dir):
    rgb_files = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.jp2')]
    chm_files = [os.path.join(chm_dir, f) for f in os.listdir(chm_dir) if f.endswith('.tif')]

    rgb_dict = {}
    chm_dict = {}

    for f in rgb_files:
        identifier = os.path.splitext(os.path.basename(f))[0]
        rgb_dict[identifier] = f

    for f in chm_files:
        identifier = os.path.basename(f)[4:-9]
        chm_dict[identifier] = f

    file_pairs = [(rgb_dict[id], chm_dict[id]) for id in rgb_dict.keys() if id in chm_dict]

    return file_pairs[:5]