import pickle as pkl


def merge_all_pkl(pkl_list, out_pkl):
    with open(out_pkl, 'wb') as out_fh:
        all_data = []
        for i_pkl in pkl_list:
            with open(i_pkl, 'rb') as i_fh:
                i_data = pkl.load(i_fh)
                all_data.extend(i_data)
        pkl.dump(all_data, out_fh)


pkl_files = ['./data_s/query_res_{}.pkl'.format(c) for c in range(0,300) ]
print(pkl_files)
merge_all_pkl(pkl_files, './data_s/all_query_res_nonsorted.pkl')
