import pymol as py
import csv
import src.util.interpUtils as iu
import os
import glob
import re
import shutil
import time

#Creates mutant for hotspot dataset given the info from the csv.
def create_mut_list_hotspot(muts_from_csv):
    mut_list = []
    for m in muts_from_csv:
        pdb4, chain, resnum, resname = m
        mut_list.append([[(resnum, chain, resname, 'ALA')], []])
    return mut_list

def read_hotspot_csv(csv_file):
    rows = []
    muts_from_csv = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    pdb4 = ''
    for i in range(1, len(rows)): #skip header row
        if rows[i][0] != '':
            code4 = rows[i][0]
        muts_from_csv.append([code4, rows[i][1], rows[i][2], rows[i][3]])
    # print muts_from_csv
    # print len(muts_from_csv)
    return muts_from_csv, rows

#Ends pymol session.
def terminate():
    py.cmd.quit()

#Housekeeping for initiating a pymol session.
def initiate():
    import __main__                       
    __main__.pymol_argv = ['pymol', '-qc']
    py.finish_launching() 
    py.cmd.set('group_auto_mode', 2)

#Uses the pymol API to mutate the pdb files.
def mutate_res(from_res, to_res):
    py.cmd.wizard("mutagenesis")
    py.cmd.do("refresh_wizard")
    print to_res
    print from_res
    py.cmd.get_wizard().set_mode(to_res)
    py.cmd.get_wizard().do_select(from_res)
    py.cmd.frame(1) #Selects the most probably rotamer
    py.cmd.get_wizard().apply()
    py.cmd.wizard(None)

def pymol_mutate(muts_from_csv, mut_list):
    mut_dir = '/scratch/PI/rondror/jlalouda/hotspot'
    test_dir = '/scratch/PI/rondror/jlalouda/muts2'
    assert len(muts_from_csv) == len(mut_list) #Sanity Check
    counter = 0
    initiate()
    pdb4, r_file, l_file, file = '', '', '', ''
    for i in range(len(muts_from_csv)):
        if pdb4 != muts_from_csv[i][0] and len(muts_from_csv[i][0]) == 4:
            print 'Done mutating ' + pdb4
            files = glob.glob(os.path.join(mut_dir, muts_from_csv[i][0], '*_cleaned.pdb'))
            r_file = [f for f in files if '_r_' in f][0]
            l_file = [f for f in files if '_l_' in f][0]
        pdb4, chain, resnum, resname = muts_from_csv[i]
        new_dir = os.path.join(mut_dir, pdb4, 'mut' + resnum)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        if re.search('_.*' + chain + '.*_', r_file) is not None:
            file = r_file
            shutil.copyfile(l_file, os.path.join(test_dir, os.path.basename(l_file)[:-4] + '_mut_' + str(i) + '.pdb'))
        else:
            file = l_file
            shutil.copyfile(r_file, os.path.join(test_dir, os.path.basename(r_file)[:-4] + '_mut_' + str(i) + '.pdb'))
        py.cmd.reinitialize()
        py.cmd.load(file)
        basename = os.path.basename(file)
        mutate_res('/' + basename[:-4] + '//' + chain + '/' + resnum, 'ALA')
        py.cmd.save(os.path.join(test_dir, basename[:-4] + '_mut_' + str(i) + '.pdb'))
        counter += 1
        # with open(os.path.join(new_dir, 'mut.csv'), 'w') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     writer.writerows([mut_list[i]])
        print 'Done with ' + file
        time.sleep(.5) #Maybe a little overkill. Pymol segfaults if things move too fast.
    terminate()
    print 'Done mutating all files'
    print str(counter) + ' files mutated.'

def main():
    muts_from_csv, rows = read_hotspot_csv('/home/users/jlalouda/hotspot_corrected.csv')
    mut_list = create_mut_list_hotspot(muts_from_csv)
    pymol_mutate(muts_from_csv, mut_list)

main()
