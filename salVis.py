import numpy as np
import pymol as py
import src.util.interpUtils as iu

def hide_all_by_model(file_name):
    py.cmd.hide('everything', 'model ' + file_name[:-4])

def get_sel_string(file_name, atom_map, negate=False):
    sel_str = ''
    sel_str += 'model ' + file_name[:-4]
    if negate:
        sel_str += ' and not ('
    else:
        sel_str += ' and ('
    for a in atom_map:
        sel_str += 'id ' + str(a) + ' or '
    sel_str = sel_str[:-4] + ')'
    return sel_str

def original_color(max_grad, atom_map, file_name, object_name, threshold=.1):
    for a in atom_map:
        grad = atom_map[a].grad
        ratio = max(grad / max_grad, threshold)
        elem = atom_map[a].elem
        color = file_name[:-4] + '_' + elem + '_' + str(a)
        if elem == 'C':
            py.cmd.set_color(color, [0.0, ratio, 0.0])
        elif elem == 'O':
            py.cmd.set_color(color, [ratio, 0.0, 0.0])
        elif elem == 'N':
            py.cmd.set_color(color, [0.0, 0.0, ratio])
        else:
            py.cmd.set_color(color, [ratio, ratio, 0.0])
        py.cmd.color(color, 'id ' + str(a) + ' and byobject ' + object_name)

def sphere_radius(max_grad, atom_map, file_name, object_name):
    for a in atom_map:
        grad = atom_map[a].grad
        ratio = min(grad / max_grad, .5)
        py.cmd.set('sphere_scale', ratio, selection='id ' + str(a) + ' and byobject ' + object_name)

def opacity(max_grad, atom_map, file_name, object_name):
    for a in atom_map:
        grad = atom_map[a].grad
        ratio = grad / max_grad
        ratio = min(.8, -ratio + 1)
        py.cmd.set('sphere_transparency', ratio, selection='id ' + str(a) + ' and byobject ' + object_name)

def update_bvalue(max_grad, atom_map, file_name, sele):
    py.cmd.alter(sele, 'b=%s'%0)
    for a in atom_map:
        grad = atom_map[a].grad
        ratio = grad / max_grad
        py.cmd.alter('(id ' + str(a) + ') and ' + sele, 'b=%s'%ratio)

def show_b(max_grad, atom_map, file_name, object_name):
    if 'receptor' in object_name:
        py.cmd.spectrum("b", "blue_white_red", 'byobject ' + object_name, minimum=0, maximum=max_grad)
    else:
        py.cmd.spectrum("b", "magenta_white_yellow", 'byobject ' + object_name, minimum=0, maximum=max_grad)
    py.cmd.recolor()

def create_object(file_name, atom_map, sele, method):
    if 'focus' in sele:
        py.cmd.select(sele, get_sel_string(file_name, atom_map, negate=False))
    else:
        py.cmd.select(sele, get_sel_string(file_name, atom_map, negate=True))
        if 'ligand' in sele:
            py.cmd.color('grey', sele)
        else:
            py.cmd.color('zinc', sele)
    py.cmd.show('cartoon', sele)

def remove_extra_structure(file_name):
    py.cmd.delete(file_name[:-4])

def robin(max_grad, atom_map, file_name, object_name, pdb_map, Ronly, sigstr):
    imp_res_num = iu.get_imp_res_max(atom_map, pdb_map)
    for a in atom_map:
        resnum = pdb_map[a].resnum
        selection = '(id ' + str(a) + ') and ' + object_name
        if resnum in imp_res_num:
            if not pdb_map[a].is_backbone_draw() or not Ronly:
                grad = atom_map[a].grad
                py.cmd.show('sticks', selection=selection)
                sphere_scale = .2
                if 'sphere' in sigstr:
                    sphere_scale = min(grad / max_grad, .5)
                if 'label' in sigstr:
                    py.cmd.label(selection, 'b')
                py.cmd.set('sphere_scale', sphere_scale, selection=selection)
                py.cmd.show('spheres', selection=selection)
        else:
            if not pdb_map[a].is_backbone_line_draw():
                py.cmd.hide('lines', selection='(id ' + str(a) + ') and ' + object_name)
    imp_res_num = [str(n[1]) for n in imp_res_num]
    imp_selection = '(resi ' + ' or resi '.join(imp_res_num) + ') and ' + object_name
    print imp_selection
    py.cmd.select('imp_' + object_name, selection=imp_selection)
    # if 'color' in sigstr:
        # py.cmd.spectrum('b', 'rainbow', 'byobject ' + object_name, minimum=0, maximum=max_grad)
    if 'surface' in sigstr:
        py.cmd.create('surf_' + object_name, 'imp_' + object_name)
        py.cmd.spectrum('b', 'rainbow', 'surf_' + object_name, minimum=0, maximum=max_grad)
        py.cmd.hide('everything', 'surf_' + object_name)
        py.cmd.show('surface', 'surf_' + object_name)
        py.cmd.set('transparency', .6)
        py.cmd.set('solvent_radius', .1)
        py.cmd.set('surface_quality', 1)

def create_molecule(data, name, method):
    py.cmd.load(iu.get_pdb_file(data['file']))
    hide_all_by_model(data['file'])
    create_object(data['file'], data['atom_map'], name + '_focus', method)
    create_object(data['file'], data['atom_map'], name + '_extra', method)
    # remove_extra_structure(data['file'])
    update_bvalue(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    if method == 'original':
        original_color(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    elif method == 'sphere_scale':
        sphere_radius(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    elif method == 'opacity':
        opacity(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    elif method == 'both':
        sphere_radius(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
        opacity(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    elif method == 'show_b':
        show_b(data['max_grad'], data['atom_map'], data['file'], name + '_focus')
    elif method == 'robin':
        pdb_map, _ = iu.get_pdb_atom_map(iu.get_pdb_file(data['file']), data['res_num'])
        robin(data['max_grad'], data['atom_map'], data['file'], name + '_focus', pdb_map, False, 'surfacelabel')
    select_AA(data['res_num'], data['res'] + data['res_num'], name + '_focus')
    select_close(data['res'] + data['res_num'], name + '_focus')

def select_AA(resi, name_sele, name_obj):
    sel_str = '(resi ' + resi + ' and ' + name_obj + ') or (resi ' + resi + ' and byobject imp_' + name_obj + ')'
    py.cmd.select(name_sele, sel_str)

def select_close(AA_sele, name_obj):
    py.cmd.select('near' + AA_sele, '(br. all within 5 of ' + AA_sele + ') and (byobject ' + name_obj + ')')

def create_pymol_session(file=None, data=None, sess_name='saliency_test', method='original',\
    example_num=0, num_rotations=20, average_rotations=True):
    assert (file is not None or data is not None)
    if file is not None:
        data = np.load(file).item()
    if average_rotations:
        l_data, r_data = iu.extract_data_across_rotations(data, example_num, num_rotations)
    else:
        l_data, r_data = iu.extract_data(data, example_num)
    py.cmd.reinitialize()
    create_molecule(l_data, 'ligand_' + l_data['file'][:4], method)
    create_molecule(r_data, 'receptor_' + r_data['file'][:4], method)
    py.cmd.cartoon("tube")
    py.cmd.set("cartoon_tube_radius", ".2")
    py.cmd.bg_color(color="grey90")
    py.cmd.save(sess_name)

def terminate():
    py.cmd.quit()

def initiate():
    import __main__                       
    __main__.pymol_argv = ['pymol', '-qc']
    py.finish_launching() 
    py.cmd.set('group_auto_mode', 2)

def main():
    initiate()       
    create_pymol_session(file='/home/users/jlalouda/surfacelets/experiments/saliency_exp3_integrated.npy',\
        sess_name='~/surfacelets/robinsMethod1AVEbackbone.pse', method='robin', average_rotations=True)
    terminate()

if __name__ == "__main__":
    main()
