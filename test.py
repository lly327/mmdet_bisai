import pickle


def load_data_list():
        ann_file = 'data/ann.pkl'
        with open(ann_file, 'rb') as f:
            ann_list = pickle.load(f)

        data_infos = []
        for i, (k,v) in enumerate(ann_list.items()):
            
            width = v['w']
            height = v['h']
            instances = []

            for anns in v['b']:
                instance = {}
                instance['bbox'] = list(anns)
                instance['bbox_label'] = 0
                instance['bbox_whs'] = (width, height)
                instance['bbox_pts'] = list(anns)
                instances.append(instance)

            data_infos.append(
                dict(
                    img_path=k,
                    img_id=i,
                    width=width,
                    height=height,
                    instances=instances
                ))

        return data_infos