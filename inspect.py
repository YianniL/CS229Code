import glob
import os
import h5py
import numpy as np
import time
import csv
from collections import Counter, defaultdict
import dill
import sys

import src.learning.data.subgrid_generation as sg
import src.learning.interaction.model as im
import src.learning.interaction.model_params as ip
import src.learning.interaction.train as tr
import src.learning.interaction.pair_to_tfrecord as ptt
import src.learning.interaction.test_params as tp
import src.learning.interaction.analyzeRotations as ar
import src.learning.interaction.analyzeDistances as ad
import src.learning.interaction.analyzeAA as aa
import src.viz.salVis as sv
import src.mutation.mutate as mu
import src.util.interpUtils as iu

def add_inspect_parser(subparsers, pp):
    iip = subparsers.add_parser(
        'inspect', description='Model inspector.',
        help='Inspect interactions from trained model', parents=[pp])
    iip.set_defaults(func=inspect)
    iip.add_argument(metavar='command', dest='command', type=str, help='which function to run')
    iip.add_argument(metavar='model.json', dest='model_json', type=str,
        help='location of model param file.')
    iip.add_argument(metavar='test.json', dest='test_json', type=str,
        help='location of training param file.')
    iip.add_argument(metavar='model_dir', dest='model_dir', type=str,
            help='location to get model files from.')
    iip.add_argument("--write_dest", help="place to write to (can be folder or file depending on what function requires)")
    iip.add_argument("--read_dest", help="place to read from")
    iip.add_argument("--pdb_code", help="pdb code for a specific residue you want to look at")
    iip.add_argument("--num_to_do", help="how many examples you want to look at", type=int)
    iip.add_argument("--mut_dir", help="current workaround for hotspot mutations")
    iip.add_argument("--hotspot_dir", help="current workaround for hotspot mutations")

def inspect(args):
    """Inspect model."""
    params = tp.load_params(args.test_json)
    model_params = ip.load_params(args.model_json)
    model_dir = args.model_dir
    command = args.command
    write_dest = args.write_dest
    read_dest = args.read_dest
    pdb_code = args.pdb_code
    num_to_do = args.num_to_do
    mut_dir = args.mut_dir
    hotspot_dir = args.hotspot_dir
    if command == 'filters':
        inspect_filters(params, model_dir, model_params, write_dest)
    elif command == 'saliencies':
        inspect_all_saliencies(params, model_dir, model_params, write_dest)
    elif command == 'rotation':
        inspect_all_rotations(params, model_dir, model_params)
    elif command == 'create_mut':
        create_mutations(params, model_dir, model_params, write_dest, read_dest, pdb_code)
    elif command == 'inspect_mut':
        inspect_mutations(params, model_dir, model_params, pdb_code, write_dest)
    elif command == 'data_stats':
        compute_data_stats(params, model_dir, model_params, write_dest)
    elif command == 'hotspot':
        print 'command interpreted properly'
        inspect_hotspot_mutations(params, model_dir, model_params, write_dest, mut_dir, hotspot_dir)
    elif command == 'distances':
        inspect_distances(params, model_dir, model_params, num_to_do)
    elif command == 'conv':
        extract_conv_layers(params, model_dir, model_params, write_dest)
    # elif command == 'imp_res': #Not ready
    #     inspect_imp_res_count(params, model_dir, model_params, num_to_do)
    else:
        print 'Command not recognized! Valid commands are: filters, saliency, rotation, create_mut, inspect_mut, and data_stats'

#Helper function that takes in various parameters and outputs the model instance
#and an iterator to loop through the data once.
def set_up_model(params, model_dir, model_params):
    if not os.path.exists(model_dir):
        print "Model dir does not exist"
        print model_dir
        return

    # Specify featurizer.
    gen = sg.TFSubgridGenerator(
            model_params, params['num_directions'], 
            1)
    # Create data iterator.
    tfrecords = glob.glob(params['dataset']) #change back to dataset
    dataset, num_batches_testing = ptt.create_tf_dataset(
        params, gen.get_gridded_pair)
    iterator = dataset.make_one_shot_iterator()

    # Initialize model.
    model = im.InteractionModel(model_dir)
    
    # If you want to feed in the grid later instead.
    model.load(params['towers'], dtypes=dataset.output_types)

    return model, iterator

def check_and_add(ex, tup):
    if tup not in ex:
        ex[tup] = True

def extract_conv_layers(params, model_dir, model_params, write_dest):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    with open('/scratch/PI/rondror/jlalouda/hotspot_corrected.csv', 'r') as f:
        reader = csv.reader(f)
        lines = []
        for r in reader:
            lines.append(r)
        data = []
        currProt = None
        for i in range(1, len(lines)):
            if lines[i][0] != currProt and lines[i][0] != '':
                currProt = lines[i][0]
            data.append((currProt, lines[i][1], lines[i][2]))
    print len(data)
    # neg_ex_l = {}
    # pos_ex_l = {}
    # neg_ex_r = {}
    # pos_ex_r = {}
    print ('3BK3', 'C', '18') in data
    counter = 0
    ex_20 = dict()
    with open('/scratch/PI/rondror/jlalouda/hotspot_saved_grids_and_convs/ex_20.dill', 'r') as in_strm:
        ex_20 = dill.load(in_strm)
    ex_20_2 = dict()
    print len(ex_20)
    for k in data:
        if k not in ex_20:
            print k[0]
    while True:
        try:
            val = model.sess.run(next_el)
            # if 'mut' in val['pdb_name'][0][0]:
            #     continue
            fc_l = model['TOWER0/base_networks/fc0/fcrelu:0']
            fc_r = model['TOWER0/base_networks/fc0_1/fcrelu:0']
            feed_dict = make_feed_dict(model, val)
            left_lin, right_lin = model.sess.run([fc_l, fc_r], feed_dict)
            print len(ex_20_2), val['pdb_name'][0][0]
            pdb = val['pdb_name'][0][0][:4]
            chain_l_pos = val['chain'][0][0].strip()
            chain_r_pos = val['chain'][0][1].strip()
            resnum_l_pos = val['residue'][0][0].strip()
            resnum_r_pos = val['residue'][0][1].strip()
            if (pdb, chain_l_pos, resnum_l_pos) in data and (pdb, chain_l_pos, resnum_l_pos) not in ex_20 and (pdb, chain_l_pos, resnum_l_pos) not in ex_20_2:
                # checked[(pdb, chain_l_pos, resnum_l_pos)] = True
                ex_20_2[(pdb, chain_l_pos, resnum_l_pos)] = ((pdb, chain_l_pos, resnum_l_pos), left_lin[0:20,:])
            if (pdb, chain_r_pos, resnum_r_pos) in data and (pdb, chain_r_pos, resnum_r_pos) not in ex_20 and (pdb, chain_r_pos, resnum_r_pos) not in ex_20_2:
                # checked[(pdb, chain_r_pos, resnum_r_pos)] = True
                ex_20_2[(pdb, chain_r_pos, resnum_r_pos)] = ((pdb, chain_r_pos, resnum_r_pos), right_lin[0:20,:])
            for i in range(20, 220, 20):
                chain_l_neg = val['chain'][i][0].strip()
                chain_r_neg = val['chain'][i][1].strip()
                resnum_l_neg = val['residue'][i][0].strip()
                resnum_r_neg = val['residue'][i][1].strip()
                chain_l_neg = val['chain'][i][0].strip()
                chain_r_neg = val['chain'][i][1].strip()
                resnum_l_neg = val['residue'][i][0].strip()
                resnum_r_neg = val['residue'][i][1].strip()
                if (pdb, chain_l_neg, resnum_l_neg) in data and (pdb, chain_l_neg, resnum_l_neg) not in ex_20 and (pdb, chain_l_neg, resnum_l_neg) not in ex_20_2:
                    # checked[(pdb, chain_l_neg, resnum_l_neg)] = True
                    ex_20_2[(pdb, chain_l_neg, resnum_l_neg)] = ((pdb, chain_l_neg, resnum_l_neg), left_lin[i:i+20,:])
                if (pdb, chain_r_neg, resnum_r_neg) in data and (pdb, chain_r_neg, resnum_r_neg) not in ex_20 and (pdb, chain_r_neg, resnum_r_neg) not in ex_20_2:
                    # checked[(pdb, chain_r_neg, resnum_r_neg)] = True
                    ex_20_2[(pdb, chain_r_neg, resnum_r_neg)] = ((pdb, chain_r_neg, resnum_r_neg), right_lin[i:i+20,:])

            # print left_lin, right_lin
            # break
            # print pdb, chain_l_pos, resnum_l_pos
            # print pdb, chain_r_pos, resnum_r_pos
            # print tuple([pdb, chain_l_pos, resnum_l_pos]) not in ex
            # print tuple([pdb, chain_r_pos, resnum_r_pos]) not in ex
            # print tuple([pdb, chain_l_pos, resnum_l_pos]) in data
            # print tuple([pdb, chain_r_pos, resnum_r_pos]) in data
            # print tuple([pdb, chain_r_pos, resnum_r_pos])
            # print tuple([pdb, chain_l_pos, resnum_l_pos])
            # print len(ex)
            # print len(ex)
            # if (pdb, chain_l_pos, resnum_l_pos) in data:
            #     checked[(pdb, chain_l_pos, resnum_l_pos)] = True
            # if (pdb, chain_r_pos, resnum_r_pos) in data:
            #     checked[(pdb, chain_r_pos, resnum_r_pos)] = True
            # if len(ex_20) == len(data):
            #     break
            # if (pdb, chain_l_pos, resnum_l_pos) not in ex and (pdb, chain_l_pos, resnum_l_pos) in data:
            #     # print 'hi1'
            #     # ex[(pdb, chain_l_pos, resnum_l_pos)] = [fc_l[0, :], val['grid'][0,0,:,:,:,:]]
            #     counter += 1
            #     ex[(pdb, chain_l_pos, resnum_l_pos)] = [left_lin[0, :], 'l_pos']
            # if (pdb, chain_r_pos, resnum_r_pos) not in ex and (pdb, chain_r_pos, resnum_r_pos) in data:
            #     # print 'hi2'
            #     ex[(pdb, chain_r_pos, resnum_r_pos)] = [right_lin[0, :], 'r_pos']
            #     # ex[(pdb, chain_r_pos, resnum_r_pos)] = [fc_r[0, :], val['grid'][0,1,:,:,:,:]]
            # if (pdb, chain_l_neg, resnum_l_neg) not in ex and (pdb, chain_l_neg, resnum_l_neg) in data:
            #     # print 'hi3'
            #     ex[(pdb, chain_l_neg, resnum_l_neg)] = [left_lin[20, :], 'l_neg']
            #     # ex[(pdb, chain_l_neg, resnum_l_neg)] = [fc_l[20, :], val['grid'][20,0,:,:,:,:]]
            # if (pdb, chain_r_neg, resnum_r_neg) not in ex and (pdb, chain_r_neg, resnum_r_neg) in data:
            #     # print 'hi4'
            #     ex[(pdb, chain_r_neg, resnum_r_neg)] = [right_lin[20, :], 'r_neg']
            #     # ex[(pdb, chain_r_neg, resnum_r_neg)] = [fc_r[20, :], val['grid'][20,1,:,:,:,:]]
            # if len(ex) == len(data):
            #     break
        except tf.errors.OutOfRangeError:
            print 'Out of files.'
            print >> sys.stderr, 'done'
            break
    # print checked
    print len(ex_20_2)
    print len(ex_20)
    # print ex
    for k in ex_20_2:
        if k not in ex_20:
            print k
            ex_20[k] = ex_20_2[k]
    print len(ex_20)
    # print len(ex)
    def save_stuff(name, stuff):
        with open(os.path.join('/scratch/PI/rondror/jlalouda/hotspot_saved_grids_and_convs', name), 'w') as f:
            dill.dump(stuff, f)
    save_stuff('ex_20.dill', ex_20)
    # save_stuff('neg_r.dill', neg_ex_r)
    # save_stuff('pos_l.dill', pos_ex_l)
    # save_stuff('pos_r.dill', pos_ex_r)
    print 'All done'


#Loops through the interacting AA for a given protein (modify code4 below) and
#creates the mutation and initial probability files for the mutants of this
#protein.
#Consider Making a mutant iterator that only loops through positive examples!
def create_mutations(params, model_dir, model_params, write_dest, read_dest, pdb_code):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    l_counter = Counter()
    r_counter = Counter()
    scores = []
    init_prob = 0.0
    num_pairs = 0
    while True:
        try:
            val = model.sess.run(next_el)
            if val['pdb_name'][0][0][:4] != pdb_code: 
                continue
            print val['pdb_name'][0][0][:4]
            data = inspect_saliency(model, val, None, integrated=True, save_flag=False, prob=True)
            print data['label']
            print data['resname'][0][0], ' ', data['resname'][0][1]
            l_data, r_data = iu.extract_data_across_rotations(data, 0, 20)
            l_atom_map = l_data['atom_map']
            r_atom_map = r_data['atom_map']
            l_pdb_map, _ = iu.get_pdb_atom_map(iu.get_pdb_file(l_data['file']), l_data['res_num'])
            r_pdb_map, _ = iu.get_pdb_atom_map(iu.get_pdb_file(r_data['file']), r_data['res_num'])
            l_imp_res_num = iu.get_imp_res_max(l_atom_map, l_pdb_map)
            r_imp_res_num = iu.get_imp_res_max(r_atom_map, r_pdb_map)
            print data['prob']
            init_prob += data['prob']
            num_pairs += 1
            for n in l_imp_res_num:
                l_counter[n] += 1
            for n in r_imp_res_num:
                r_counter[n] += 1
        except tf.errors.OutOfRangeError: 
            break
    print init_prob
    print num_pairs
    init_prob /= num_pairs
    print init_prob
    print l_counter
    print r_counter
    with open(write_dest, 'wb') as f:
        dill.dump({'prob':init_prob, 'l':l_counter, 'r':r_counter}, f)

    # with open(read_dest, 'rb') as f: #Run the program in two parts to get the intermediate dill file.
    #     temp = dill.load(f)
    #     l_counter = temp['l']
    #     r_counter = temp['r']
    #     init_prob = temp['prob']
    # l_pdb_map = iu.get_pdb_res_map(iu.get_pdb_file(l_data['file']))
    # r_pdb_map = iu.get_pdb_res_map(iu.get_pdb_file(r_data['file']))
    # mut_list = mu.create_mut_list(l_counter, r_counter, l_pdb_map, r_pdb_map, True)
    # files = [l_data['file'], r_data['file']]
    # print mut_list[0]
    # mu.mutate_all_res(files, mut_list=mut_list, init_prob=init_prob)

#Helper function to deal with the weird folder situation for hotspot mutate.
def get_full_file_name(f, mut_dir, hotspot_dir):
    if 'mut' in f:
        return os.path.join(mut_dir, f)
    else:
        return os.path.join(hotspot_dir, f[:4], f)

def get_code(f):
    if 'mut' in f:
        return f[:f.find('_')] + f[f.rfind('_') + 1:-4]
    else:
        return f[:f.find('_')]

def inspect_hotspot_mutations(params, model_dir, model_params, write_dest, mut_dir, hotspot_dir):
    print "in inspect hotspot mutations"
    print >> sys.stderr, 'in hotspot'
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    counter = 0
    not_counter = 0
    dict_base = os.path.basename(params['keep_file'])[:-4]
    surf_probs = []
    l_imp_dict = defaultdict(int)
    r_imp_dict = defaultdict(int)
    while True:
        try:
            val = model.sess.run(next_el)
            file_code = get_code(val['pdb_name'][0][0])
            print counter, not_counter, file_code
            print >> sys.stderr, str(counter) + ' ' + str(not_counter) + ' ' + file_code
            if dict_base != file_code:
                not_counter += 1
                continue
            data = inspect_saliency(model, val, None, integrated=True, save_flag=False, prob=True)
            l_data, r_data = iu.extract_data_across_rotations(data, 0, 20) #Assumes 0th index is start of the positive
            l_atom_map = l_data['atom_map']
            r_atom_map = r_data['atom_map']
            l_pdb_map, _ = iu.get_pdb_atom_map(get_full_file_name(l_data['file'], mut_dir, hotspot_dir), l_data['res_num'])
            r_pdb_map, _ = iu.get_pdb_atom_map(get_full_file_name(r_data['file'], mut_dir, hotspot_dir), r_data['res_num'])
            l_imp_res_num = iu.get_imp_res_max(l_atom_map, l_pdb_map)
            r_imp_res_num = iu.get_imp_res_max(r_atom_map, r_pdb_map)
            surf_probs.append(data['prob'])
            for res_tup in l_imp_res_num:
                l_imp_dict[res_tup] += 1
            for res_tup in r_imp_res_num:
                r_imp_dict[res_tup] += 1
            counter += 1
        except tf.errors.OutOfRangeError:
            break
            print 'Out of files.'
            print >> sys.stderr, 'done'
    print counter
    print not_counter
    print >> sys.stderr, str(counter) + ' ' + str(not_counter)
    with open(os.path.join(write_dest, dict_base + '_surf_probs.dill'), 'w') as f1:
        dill.dump(surf_probs, f1)
    with open(os.path.join(write_dest, dict_base + '_l_imp_dict.dill'), 'w') as f2:
        dill.dump(l_imp_dict, f2)
    with open(os.path.join(write_dest, dict_base + '_r_imp_dict.dill'), 'w') as f3:
        dill.dump(r_imp_dict, f3)
        
#Once the mutations have been created, gets final interface probability and
#appends it to each mutants probability file.
def inspect_mutations(params, model_dir, model_params, pdb_code, write_dest='/scratch/PI/rondror/ppi/benchmark5/cleaned/mutations/'):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    num_pairs = 0
    mut_prob = 0
    mut_num = None
    counter = 0
    while True:
        try:
            val = model.sess.run(next_el)
            file_name = val['pdb_name'][0][0]
            print file_name
            if mut_num is not None and mut_num != file_name[file_name.rfind('_')+1:-4]: #Done with the mutant
                new_dir = os.path.join(write_dest, pdb_code, 'mut' + str(mut_num))
                mut_prob /= num_pairs
                with open(os.path.join(new_dir, 'prob.csv'), 'ab') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow([mut_prob])
                print counter
                print file_name
                print num_pairs
                num_pairs = 0
                mut_prob = 0
                counter += 1
            mut_num = file_name[file_name.rfind('_')+1:-4]
            mut_prob += get_probs(model, feed_dict=make_feed_dict(model, val), num_rotations=20, val=val)
            num_pairs += 1
        except tf.errors.OutOfRangeError: #Gotta catch 'em all!
            new_dir = os.path.join(write_dest, pdb_code, 'mut' + str(mut_num))
            #mut_prob /= num_pairs
           # with open(os.path.join(new_dir, 'prob.csv'), 'ab') as csvfile:
           # print num_pairs
            break
    print counter
    print 'Done inspecting mutations for ' + pdb_code

#Retrieves the filter value for the first conv layer and saves it to the
#hardcoded file below.
def inspect_filters(params, model_dir, model_params, write_dest):
    model, iterator = set_up_model(params, model_dir, model_params)
    
    # If you want to feed in the grid later instead.
    model.load(1, has_seq=False, iterator=None)

    weights_map = {}

    layer1_weights_tensor = model["base_networks/conv0/weights:0"]
    weights_map['layer1_weights'] = model.sess.run(layer1_weights_tensor)

    layer1_biases_tensor = model["base_networks/conv0/biases:0"]
    weights_map['layer1_biases'] = model.sess.run(layer1_biases_tensor)
    np.save(write_dest, weights_map)

#Gets the res counts for the train/test sets and inter/noninter pairs
#and saves it to the hardcoded files below.
def compute_data_stats(params, model_dir, model_params, write_dest):
    model, iterator = set_up_model(params, model_dir, model_params)

    # Get batch
    next_el = iterator.get_next()

    # Loop through the different examples.
    res_counts_inter = {}
    res_counts_no_inter = {}
    tic = time.time()
    while True:
        try:
            example = model.sess.run(next_el)
            for i in range(0, example['label'].shape[0],
                    params['num_directions']):
                res = example['resname'][i][0] + example['resname'][i][1]
                if example['label'][i] == 1:
                    if res in res_counts_inter:
                           res_counts_inter[res] += 1
                    else:
                           res_counts_inter[res] = 1
                else:
                    if res in res_counts_no_inter:
                        res_counts_no_inter[res] += 1
                    else:
                        res_counts_no_inter[res] = 1
        except tf.errors.OutOfRangeError: 
            break
    np.save(os.path.join(write_dest, 'res_counts_inter_train.npy'), res_counts_inter)
    np.save(os.path.join(write_dest, 'res_counts_no_inter_train.npy'), res_counts_no_inter)
    toc = time.time()
    print str(toc - tic) + ' seconds elapsed'
    print "All done, res counts saved."

#Sees to what degree the important atoms are the same across rotations. Maybe
#more relevant to do this for residues instead.
def inspect_all_rotations(params, model_dir, model_params, num_to_do=100):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    tic = time.time()
    l_av_sim = []
    r_av_sim = []
    counter = 0
    while True:
        try:
            val = model.sess.run(next_el)
            data = inspect_saliency(model, val, None, integrated=True)
            l_sims, r_sims = analyzeRotations(data=data, num_rotations=params['num_directions'], single=False)
            l_av_sim.append(l_sims)
            r_av_sim.append(r_sims)
            counter += 1
            if counter == num_to_do:
                break
        except tf.errors.OutOfRangeError: 
            break
    print l_av_sim
    print r_av_sim
    print len(l_av_sim)
    print len(r_av_sim)
    print 'Average for ligands.'
    print np.mean(np.asarray(l_av_sim))
    print 'Average for receptors.'
    print np.mean(np.asarray(r_av_sim))
    toc = time.time()
    print 'All done inspecting rotations.'
    print str(toc - tic) + ' seconds elapsed'

#Sees how close the important atoms are to the CA of the residue of interest.
def inspect_distances(params, model_dir, model_params, num_to_do=100):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    tic = time.time()
    counter = 0
    while True:
        try:
            val = model.sess.run(next_el)
            data = inspect_saliency(model, val, integrated=True)
            counter += 1
            if counter == num_to_do:
                break
        except tf.errors.OutOfRangeError: 
            break
    toc = time.time()
    print 'All done inspecting distances.'
    print str(toc - tic) + ' seconds elapsed'

#Gets a count of the AA type for the important residues.
def inspect_imp_res_count(params, model_dir, model_params, num_to_do=100):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    tic = time.time()
    res_counts_pos = Counter()
    res_counts_neg = Counter()
    counter = 0
    while True:
        try:
            val = model.sess.run(next_el)
            data = inspect_saliency(model, val, integrated=True)
            aa.analyze_aa(res_counts_pos, example_num=0, data=data)
            aa.analyze_aa(res_counts_neg, example_num=20, data=data)
            counter += 1
            if counter == num_to_do:
                break
        except tf.errors.OutOfRangeError: 
            break
    toc = time.time()
    print 'All done inspecting distances.'
    print str(toc - tic) + ' seconds elapsed'

#Creates pymol sessions highlighting the important residues.
def inspect_all_saliencies(params, model_dir, model_params, write_dest, num_to_do=100, make_pymol_session=True):
    import tensorflow as tf
    model, iterator = set_up_model(params, model_dir, model_params)
    next_el = iterator.get_next()
    tic = time.time()
    if make_pymol_session:
        sv.initiate()
    counter = 0
    files = []
    while True:
        try:
            val = model.sess.run(next_el)
            f = val['pdb_name'][0][0][:4]
            if files.count(f) >= 3: #Skip represented proteins
                continue
            data = inspect_saliency(model, val, write_dest, integrated=True, save_flag=False)
            print data['label']
            print data['resname'][0][0]
            files.append(f)
            if make_pymol_session:
                methods = ['robin']
                for m in methods:
                    sv.create_pymol_session(data=data, sess_name=os.path.join(write_dest,\
                     data['pdb_name'][0][0][:4] + data['resname'][0][0] + data['resname'][0][1] + 'pos' + str(counter) + '.pse'),\
                      method=m, example_num=0, average_rotations=True)
                    sv.create_pymol_session(data=data, sess_name=os.path.join(write_dest,\
                     data['pdb_name'][20][0][:4] + data['resname'][20][0] + data['resname'][20][1] + 'neg' + str(counter) + '.pse'),\
                      method=m, example_num=20, average_rotations=True)
                counter += 1
                print counter
                if counter == num_to_do:  #Generates
                    break
        except tf.errors.OutOfRangeError: 
            break
    if make_pymol_session:
        sv.terminate()
    toc = time.time()
    print 'All done inspecting saliencies.'
    print str(toc - tic) + ' seconds elapsed'

#Retrieve the total accuracy and probability of interaction for POSITIVE
#examples. ASSUMES THAT BATCH SIZE IS 1!
def get_probs(model, feed_dict=None, num_rotations=20, val=None):
    if feed_dict is None and val is not None:
        feed_dict = make_feed_dict(model, val)
    output = model.sess.run([model['out:0'], model['acc:0']], feed_dict)
    probs = output[0]
    accuracy = output[1]
    output = model.sess.run([model['out:0'], model['acc:0']], feed_dict)
    probs = output[0]
    accuracy = output[1]
    return np.mean(probs[:num_rotations])

#Helper function for passing in the left and right grids to the feed dict.
def make_feed_dict(model, val):
    feed_dict = model._get_feed_dict(False)
    feed_dict[model.data['grid'].name] = val['grid']
    feed_dict[model.data['label'].name] = val['label']
    return feed_dict

#Runs integrated gradients or vanilla gradient saliency finding. 
def inspect_saliency(model, val, write_dest, integrated=True, save_flag=False, num_rotations=20, prob=False):    
    import tensorflow as tf
    # Run model.
    # Create the feed dict
    # feed_dict = model._get_feed_dict(False)
    grid_left = val['grid'][:,0,:,:,:,:]
    grid_right = val['grid'][:,1,:,:,:,:]
    # feed_dict['grid_left:0'] = grid_left
    # feed_dict['grid_right:0'] = grid_right

    feed_dict = make_feed_dict(model, val)
    
    pred_tensor = model["TOWER0/prediction/Squeeze:0"]
    grads_tensor = tf.gradients(pred_tensor, model[model.data['grid'].name])[0]
    if prob:
        val['prob'] = get_probs(model, feed_dict, num_rotations=num_rotations)
    #This code is for integrated gradients
    if integrated:
        prev_alpha = 0.0
        integrated_left = np.zeros_like(grid_left)
        integrated_right = np.zeros_like(grid_right)
        steps = 30
        for i in range(1, steps + 1):
            alpha = (1.0/steps) * i
            feed_dict[model.data['grid'].name] = val['grid'] * alpha
            gradients = model.sess.run(grads_tensor, feed_dict)
            grads_left = gradients[:, 0, :, :, :, :]
            grads_right = gradients[:, 1, :, :, :, :]
            integrated_left += grads_left * (alpha * grid_left - prev_alpha * grid_left) 
            integrated_right += grads_right * (alpha * grid_right - prev_alpha * grid_right)
            prev_alpha = alpha
        grads_left_max = np.amax(np.absolute(integrated_left * grid_left), axis=4)
        grads_right_max = np.amax(np.absolute(integrated_right * grid_right), axis=4)
        val['grads'] = [grads_left_max, grads_right_max]
    else:
        gradients = model.sess.run([grads_left_tensor, grads_right_tensor], feed_dict)
        grads_left = gradients[0]
        grads_right = gradients[1]
        grads_left_max = np.amax(np.absolute(grads_left), axis=5)
        grads_right_max = np.amax(np.absolute(grads_right), axis=5)
        val['grads'] = [grads_left_max, grads_right_max]
    if save_flag:
        np.save(write_dest, val)
    return val
