import csv
import dill
import os

AAs = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',\
'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

def get_one_hot_AA_list(resname):
    AA_list = [0 for i in range(20)]
    for i in range(len(AA_list)):
        if resname == AAs[i]:
            AA_list[i] = 1
            break
    return AA_list

#You need to do i - 1 because we started from the header here but not previously
def read_hotspot_csv(csv_file, results_dir):
    features = []
    labels = []
    rows = []
    muts_from_csv = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    currUnmutProtein = None
    currUnmutProb = None
    currUnmutProbNum = None
    for i in range(1, len(rows)):
        print(i)
        if rows[i][0] != currUnmutProtein and rows[i][0] != '': #We have hit a new protein
            currUnmutProtein = rows[i][0]
            with open(os.path.join(results_dir, currUnmutProtein + '_surf_probs.dill'), 'r') as f:
                data = dill.load(f)
            data = [float(d) for d in data]
            currUnmutProb = sum(data) / len(data)
            currUnmutProbNum = len(data)
        currChain = rows[i][1]
        currResnum = rows[i][2]
        currResname = rows[i][3]
        label = 1 if rows[i][5] == 'HS' else 0
        labels.append([label])
        with open(os.path.join(results_dir, currUnmutProtein + str(i-1) + '_surf_probs.dill'), 'r') as f:
            data = dill.load(f)
        data = [float(d) for d in data]
        currMutProb = sum(data) / len(data)
        currMutProbNum = len(data)
        feature = get_one_hot_AA_list(currResname)
        feature.append(abs(currMutProb - currUnmutProb))
        feature.append(abs(currMutProbNum - currUnmutProbNum))
        with open(os.path.join(results_dir, currUnmutProtein + str(i-1) + '_l_imp_dict.dill'), 'r') as f:
            l_res = dill.load(f)
        with open(os.path.join(results_dir, currUnmutProtein + str(i-1) + '_r_imp_dict.dill'), 'r') as f:
            r_res = dill.load(f)
        res_tup = (currChain, int(currResnum))
        if res_tup in l_res and res_tup in r_res:
            print 'RUH ROH'
        l_sum = 0.0
        for k in l_res:
            l_sum += l_res[k]
        r_sum = 0.0
        for k in r_res:
            r_sum += r_res[k]
        else:
            if res_tup in l_res:
                # print 'GOOD L'
                print l_res[res_tup], l_sum
                feature.append(l_res[res_tup]/l_sum)
            elif res_tup in r_res:
                # print 'GOOD R'
                print r_res[res_tup], r_sum
                feature.append(r_res[res_tup]/r_sum)
            else:
                # print 'NOT FOUND'
                feature.append(0.0) #Guess it wasn't important enough...
        print feature, label
        features.append(feature)
    print len(labels), len(features)
    with open('/scratch/PI/rondror/jlalouda/hotspot_x.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(features)
    with open('/scratch/PI/rondror/jlalouda/hotspot_y.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(labels)

def main():
    read_hotspot_csv('/home/users/jlalouda/surfacelets/hotspot_corrected.csv', '/scratch/PI/rondror/jlalouda/hotspot_writedir')

main()