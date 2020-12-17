import os
import glob


def boxdataset_cfg():
    category_list = ['train', 'val']
    path_list = ['images', 'labels']

    # dataset_path init
    # https://jvvp.tistory.com/984
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # dir_list = {path_name: os.path.join(root_dir, path_name) for path_name in path_list}

    dataset_dir = {}
    for cate in category_list:  # train, val
        dataset_dir[cate] = {}
        for dir_name in path_list:     # images, labels
            dataset_dir[cate][dir_name] = os.path.join(cate, dir_name)

    return dataset_dir


def make_dataset(dataset_dir):
    # dataloader에서 사용할 text 파일 제작
    # labels : cx, cy, w, h
    ext_match = {'images': '*.jpg',
                 'labels': '*.txt'}
    dataset_path = {'train': {},
                    'val': {}}
    dataset = {'train': {},
               'val': {}}

    for cate, dir_list in dataset_dir.items():      # train, val
        for dir_name, dir_path in dir_list.items():    # images, labels
            # load dataset_path_list
            # [train,val] / [images,labels]
            dataset_path[cate][dir_name] = glob.glob(os.path.join(dir_path, ext_match[dir_name]))

            # [train,val] / [images,labels] / [filename] / file_path or [line list]
            for file_path in dataset_path[cate][dir_name]:
                file_name = file_path.split('\\')[-1] if os.name == 'nt' else file_path.split('/')[-1]
                file_name, ext = file_name.split('.')
                if file_name not in dataset[cate]:
                    dataset[cate][file_name] = {}

                if dir_name == 'images':
                    dataset[cate][file_name]['path'] = file_path
                else:
                    dataset[cate][file_name]['data'] = []
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            dataset[cate][file_name]['data'].append(line.rstrip())

    f = {
        'train': open('dataset_train.txt', 'w+'),
        'val': open('dataset_val.txt', 'w+')
    }
    for cate, file_names in dataset.items():      # train, val
        for file_name, file_data in file_names.items():    # images, labels
            file_path = file_data['path']
            for line in file_data['data']:
                f[cate].write(file_path + ' ' + line + '\n')

        f[cate].close()


if __name__ == '__main__':
    dataset_dir = boxdataset_cfg()
    print(boxdataset_cfg())
    make_dataset(dataset_dir)
