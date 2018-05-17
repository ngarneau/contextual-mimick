from intrinsic_evaluation import Evaluator
import pickle as pkl
import os

def load_evaluation(model_name):
    filepath = './intrinsic/intrinsic_{}.pkl'.format(model_name)
    with open(filepath, 'rb') as file:
        results = pkl.load(file)
    
    return results


if __name__ == '__main__':
    datasets = ['conll', 'semeval', 'sent']
    ns = [5, 9, 15]
    models = ['lrcomick_v1.2', 'comickdev_v2.1', 'lrcomickcontextonly_v1.2']

    path = './top_n/eucl_dist/'
    # path = './top_n/cos_sim/'
    os.makedirs(path, exist_ok=True)
    for dataset in datasets:
        for n in ns:
            for model in models:
                model_name = '{dataset}_n{n}_k2_d100_e100_{model}'.format(dataset=dataset, n=n, model=model)
                
                results = load_evaluation(model_name)
                # cos_sim_res = results.sort_by('cos_sim', reverse=True)
                eucl_dist_res = results.sort_by('eucl_dist')

                with open(path + model_name + '.txt', 'w', encoding='utf-8') as file:
                    row = '\t'.join(['#', 'cos_sim', 'eucl_d', 'label', 'context']) + '\n'
                    file.write(row)
                    for i, r in enumerate(eucl_dist_res):
                    # for i, r in enumerate(cos_sim_res):
                        row = [str(i), '{:.3f}'.format(r.cos_sim), '{:.2f}'.format(
                            r.eucl_dist), r.label, ' '.join(r.context)]
                        row = '\t'.join(row) + '\n'
                        file.write(row)
