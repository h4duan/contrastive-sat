import numpy as np
import copy
import random
from random import randrange
import itertools



def generate_k_iclause(n, k):
  vs = np.random.choice(n, size=min(n, k), replace=False)
  return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]



def get_num_literal(instance):
    #print(type(instance))
    flat_list = np.asarray([item for sublist in instance for item in sublist])
    sorted_literals = np.unique(np.absolute(flat_list))
    return len(sorted_literals)

def remove_space(instance):
    sorted_literals = np.unique(np.absolute(np.asarray([item for sublist in instance for item in sublist])))
    num_literals = len(sorted_literals)
    #print(sorted_literals)
    #print(instance)
    if sorted_literals[-1] == num_literals:
        return instance
        #print(sorted_literals)
    #print(np.argwhere(sorted_literals != np.arange(1, num_literals+1)))
    wrong_index = np.argwhere(sorted_literals != np.arange(1, num_literals+1))[0]
    num_wrong_literal = sorted_literals[wrong_index]
    for i in range(len(instance)):
        for j in range(len(instance[i])):
            if abs(instance[i][j]) < num_wrong_literal:
                continue
            else:
                correct_id = list(sorted_literals).index(abs(instance[i][j])) + 1
                if instance[i][j] > 0:
                    instance[i][j] = correct_id
                else:
                    instance[i][j] = -correct_id
        #:wq
    #sorted_literals = np.unique(np.absolute(np.asarray([item for sublist in instance for item in sublist])))
    #print(sorted_literals)
    #print(" ")
        #num_literals = len(sorted_literals)
    return instance



def tautology(clause):
    remove_redundant = np.absolute(np.unique(np.asarray(clause)))
    return len(remove_redundant) != len(np.unique(remove_redundant))



class Augmentation:

    def __init__(self, instance, at, cr, ve, be, sub, gcl_cla, gcl_var, gcl_link, gcl_sub, ve_small=False):
        self.instance = instance
        self.at = at
        self.cr = cr
        self.ve = ve
        self.be = be
        self.sub = sub
        self.ve_small = ve_small

        self.gcl_cla = gcl_cla
        self.gcl_var = gcl_var
        self.gcl_link = gcl_link
        self.gcl_sub = gcl_sub


    def unit_propagation(self, instance):
        def has_unit(instance):
            for i in range(len(instance)):
                if len(instance[i]) == 1:
                    # print(instance[i])
                    return True
            return False

        while has_unit(instance):
            for c in range(len(instance)):
                if len(instance[c]) != 1:
                    continue
                v = instance[c][0]
                delete_clause = []
                for j in range(len(instance)):
                    if v in instance[j]:
                        delete_clause += [j]
                    if -v in instance[j]:
                        instance[j].remove(-v)
                instance = [i for j, i in enumerate(instance) if j not in delete_clause]
                break

        return remove_space(instance)

    def pure_literal(self, instance):
        while True:
            pure_literal = []
            multi_literal = []
            for c in instance:
                for l in c:
                    v = abs(l)
                    if v in pure_literal:
                        pure_literal.remove(v)
                        multi_literal += [v]
                    elif v not in multi_literal:
                        pure_literal += [v]
            # print(pure_literal)
            if len(pure_literal) == 0:
                break
            delete_clause = []
            for i in range(len(instance)):
                for l in instance[i]:
                    v = abs(l)
                    if v in pure_literal and i not in delete_clause:
                        delete_clause += [i]
            # print(delete_clause)
            instance = [i for j, i in enumerate(instance) if j not in delete_clause]
        return remove_space(instance)


    def blocked_clause_elimination(self, instance):
        random.shuffle(instance)
        occurence_list = {}
        for i in range(len(instance)):
            for l in instance[i]:
                if l in occurence_list:
                    occurence_list[l].append(i)
                else:
                    occurence_list[l] = [i]
        blocked_clause = []
        if len(instance) == 1:
            return instance
        for i in range(len(instance)):
            # print(clause1)
            removed = get_num_literalFalse
            #blocked = True
            for l in instance[i]:
                blocked = True
                if removed:
                    break
                if -l not in occurence_list:
                    continue
                for j in occurence_list[-l]:
                        # print(l, clause1)
                    clause1 = copy.deepcopy(instance[i])
                    clause2 = copy.deepcopy(instance[j])
                    clause1.remove(l)
                    clause2.remove(-l)
                    resolvent = clause1 + clause2
                    if not tautology(resolvent):
                        blocked = False
                        break
                if blocked:
                    #instance.pop(i)
                    blocked_clause += [i]
                    removed = True
                    break
                    #return remove_space(instance)
        if len(blocked_clause) == 0:
            return remove_space(instance)
        random.shuffle(blocked_clause)
        blocked_clause= blocked_clause[:max(round(len(blocked_clause)*0.9), 1)]
        new_instance = [i for j, i in enumerate(instance) if j not in blocked_clause]
        #print("blocked claude " + str(len(blocked_clause)))
        if len(new_instance) == 0:
            index = random.randint(0, len(instance)-1)
            return remove_space([instance[index]])
        return remove_space(new_instance)

    def subsume_clause_elimination(self, instance):
        occurence_list = {}
        for i in range(len(instance)):
            for l in instance[i]:
                if l in occurence_list:
                    occurence_list[l].append(i)
                else:
                    occurence_list[l] = [i]
        subsumed = []
        for i in range(len(instance)):
            clause = instance[i]
            min_lit = clause[0]
            min_val = len(occurence_list[min_lit])
            for j in range(1, len(clause)):
                occu = len(occurence_list[clause[j]])
                if occu < min_val:
                    min_lit = clause[j]
                    min_val = occu
            for eli in occurence_list[min_lit]:
                if eli != i and eli not in subsumed:
                    if len(instance[eli]) > len(clause):
                        if set(clause).issubset(set(instance[eli])):
                            subsumed += [eli]
        #print(len(subsumed))
        if len(subsumed) == 0:
            return remove_space(instance)
        #random.shuffle(subsumed)
        #subsumed= subsumed[:max(round(len(subsumed)*0.9), 1)]
        #print("subsume " + str(len(subsumed)))
        instance = [i for j, i in enumerate(instance) if j not in subsumed]
        return remove_space(instance)


    def add_trivial(self, instance, added_literal, added_clause, p_positive=0.5):
        num_literal = get_num_literal(instance)
        num_clause = len(instance)
        num_added_literals = max(int(num_literal * added_literal), 1)
        num_added_literals = random.randint(max(int(num_added_literals * 0.5), 1), num_added_literals)
        #print(num_added_literals)
        num_added_clauses = max(int(num_clause * added_clause), 1)
        num_added_clauses = random.randint(max(int(num_added_clauses * 0.5), 1), num_added_clauses)
        #print(num_added_clauses)
        for i in range(num_added_literals):
            literal_id = num_literal + i + 1
            if random.uniform(0, 1) < p_positive:
                literal_id = -literal_id
            instance.append([literal_id])
            for _ in range(num_added_clauses):
                clause_id = randrange(len(instance))
                if literal_id in instance[clause_id] or -literal_id in instance[clause_id]:
                    continue
                instance[clause_id] = instance[clause_id] + [-literal_id]
            for _ in range(num_added_clauses):
                k_base = 1 if random.random() < 0.3 else 2
                k = k_base + np.random.geometric(0.4)
                new_clause = generate_k_iclause(num_literal+i, k)
                new_clause.append(literal_id)
                instance.append(new_clause)
            random.shuffle(instance)
        return remove_space(instance)


    def variable_elimination(self, instance, eliminate_var, max_resolvent, pos_variable=4, ve_small=False):
        #print("original " + str(len(instance)))
        v = randrange(get_num_literal(instance)) + 1

        if ve_small:
            num_variable = get_num_literal(instance)
            frequency = {}
            for clause in instance:
                for literal in clause:
                    if literal not in frequency:
                        frequency[literal] = 0
                    else:
                        frequency[literal] += 1
            max_num = 9999999
            num_added = np.zeros(num_variable)
            for i in range(num_variable):
                num_pos = 0
                num_neg = 0
                if i+1 in frequency:
                    num_pos = frequency[i+1]
                if -(i+1) in frequency:
                    num_neg = frequency[-(i+1)]
                if num_pos == 0 or num_neg == 0:
                    num_added[i] = max_num
                else:
                    num_added[i] = num_pos * num_neg - (num_pos + num_neg)
        #print(num_added)
        #print(num_added)
            indices = num_added.argsort()
        #print(num_added[indices])
            v = indices[randrange(min(pos_variable, num_variable))]

        #print(v) 
        
        #v = randrange(get_num_literal(instance))+1
        #print(v)
        #if elimiate_second:
        #    v = indices[1]
        #num_eliminate_variable = random.randint(1, max(int(num_variable * eliminate_var), 1))

        #eliminate_variables = np.random.randint(1, num_variable+1, num_eliminate_variable)
        positive_clause = []
        negative_clause = []
        for i in range(len(instance)):
            c = instance[i]
            if v in c:
                positive_clause += [i]
            if -v in c:
                negative_clause += [i]

        if len(positive_clause) == 0:
            if len(negative_clause) == len(instance):
                index = random.randint(0, len(instance)-1)
                return remove_space([instance[index]])
            instance = [i for j, i in enumerate(instance) if j not in negative_clause]
            return remove_space(instance)
        if len(negative_clause) == 0:
            if len(positive_clause) == len(instance):
                index = random.randint(0, len(instance)-1)
                return remove_space([instance[index]])
            instance = [i for j, i in enumerate(instance) if j not in positive_clause]
            return remove_space(instance)

        resolvents = []

        for pc in positive_clause:
            for nc in negative_clause:
                clause1 = copy.deepcopy(instance[pc])
                clause2 = copy.deepcopy(instance[nc])
                #if len(clause1) == 0 and len(clause2) == 0:
                    #continue
                clause1.remove(v)
                clause2.remove(-v)
                resolvent = clause1 + clause2
                if len(resolvent) == 0:
                    continue
                resolvent = list(set(resolvent))
                #print(resolvent, tautology(resolvent))
                #if tautology(resolvent):
                    #print(resolvent)
                if not tautology(resolvent):
                  resolvents.append(resolvent)
        #print("before adding resolvent " + str(len(instance)))
        #print("length of resolvents " + str(len(resolvents)))
        #if len
        if len(resolvents) <= 1000 and len(resolvent) > 0:
            indices = positive_clause + negative_clause
            instance = [i for j, i in enumerate(instance) if j not in indices]
            instance += resolvents
        else:
            return remove_space(instance)

        #print("before subsumption " + str(len(instance)))
        if ve_small:
            occurence_list = {}
            for i in range(len(instance)):
                for l in instance[i]:
                    if l in occurence_list:
                        occurence_list[l].append(i)
                    else:
                        occurence_list[l] = [i]
            subsumed = []
            for i in range(len(instance)):
                clause = instance[i]
                min_lit = clause[0]
                min_val = len(occurence_list[min_lit])
                for j in range(1, len(clause)):
                    occu = len(occurence_list[clause[j]])
                    if occu < min_val:
                        min_lit = clause[j]
                        min_val = occu
                for eli in occurence_list[min_lit]:
                    if eli != i and eli not in subsumed:
                        if len(instance[eli]) > len(clause):
                            if set(clause).issubset(set(instance[eli])):
                                subsumed += [eli]
            #print(subsumed)
            if len(subsumed) >= len(instance):
                subsumed = subsumed[:int(len(subsumed)*0.9)]
            instance = [i for j, i in enumerate(instance) if j not in subsumed]
        #print("after subsumption " + str(len(instance)))
        #new_instance = self.blocked_clause_elimination(new_instance)
        #print(len(instance))
        #print(" ")
        #instance.sort()
        #instance = list(k for k,_ in itertools.groupby(instance))


        
        return remove_space(instance)
        #return instance

    def clause_resolution(self, instance, added_prop):
        #assert(instance != None)
        num_length = len(instance)
        num_resolvents = max(int(added_prop * num_length), 1)
        num_resolvents = random.randint(max(int(num_resolvents * 0.2), 1), num_resolvents)
        for _ in range(num_resolvents):
            #if None in instance:
            #print(instance)
            random.shuffle(instance)
            #if len(instance[0]) == 0:
            #    print(instance)
            l = random.choice(instance[0])
            for c2 in range(1, len(instance)):
                if -l in instance[c2]:
                    clause1 = copy.deepcopy(instance[0])
                    clause2 = copy.deepcopy(instance[c2])
                    clause1.remove(l)
                    clause2.remove(-l)
                    resolvent = clause1 + clause2
                    if len(resolvent) == 0:
                        continue
                    resolvent = list(dict.fromkeys(resolvent))
                    if not tautology(resolvent):
                        instance.append(resolvent)
                        break
        return remove_space(instance)

    def gcl_cla_drop(self, instance, args):
        del_cl_cnt = int(np.ceil(len(instance) * args.gcl_del_clause_ratio))

        for _ in range(del_cl_cnt):
            cla_i = random.randrange(0, len(instance))
            del instance[cla_i]

        # A workaround for when instance becomes empty
        if len(instance) == 0:
            return [[1]]

        return remove_space(instance)

    def gcl_var_drop(self, instance, args):
        var_cnt = max([max(np.abs(x)) for x in instance])
        del_var_cnt = int(np.ceil(var_cnt * args.gcl_del_var_ratio))

        for _ in range(del_var_cnt):
            rand_cla = instance[random.randrange(0, len(instance))]
            rand_var = rand_cla[random.randrange(0, len(rand_cla))]

            for c_i, cla in reversed(list(enumerate(instance))):
                for v_i, var in reversed(list(enumerate(cla))):
                    if var == rand_var: # or var == -rand_var:
                        del cla[v_i]

                if len(cla) == 0:
                    del instance[c_i]
        # A workaround for when instance becomes empty
        if len(instance) == 0:
            return [[1]]

        return remove_space(instance)

    def gcl_link_purt(self, instance, args):
        link_cnt = sum([len(x) for x in instance])
        var_cnt = max([max(np.abs(x)) for x in instance])
        var_range = set(range(1, var_cnt + 1)) | set(range(-var_cnt, 0))
        link_purt_cnt = int(np.ceil(link_cnt * args.gcl_purt_link_ratio))

        for _ in range(link_purt_cnt):
            c_i = random.randrange(0, len(instance))
            rand_cla = instance[c_i]

            if random.uniform(0, 1) > 0.5:  # pick between adding or removing a link
                del rand_cla[random.randrange(0, len(rand_cla))]

                if len(rand_cla) == 0:
                    del instance[c_i]

            else:  # add a new link (i.e., a random literal to a random clause)
                if len(var_range - set(instance[c_i])) == 0: # check if the clause is maximal
                    continue
                sign = random.choice([-1, 1])
                var = random.choice(list(var_range - set(instance[c_i]))) # pick a var not already in the clause
                instance[c_i] += [sign * var]
                
        # A workaround for when instance becomes empty
        if len(instance) == 0:
            return [[1]]

        return remove_space(instance)

    def gcl_subgraph_D(self, instance, args):
        cla_cnt = len(instance)
        var_cnt = max([max(np.abs(x)) for x in instance])
        sub_node_cnt = int(np.ceil((cla_cnt + var_cnt) * (1 - args.gcl_subgraph_ratio)))

        lit_2_cla = {l: [] for l in list(range(1, var_cnt+1)) + list(range(-var_cnt, 0))}
        for cla_i, cla in enumerate(instance):
            for lit in cla:
                lit_2_cla[lit] += [cla_i]

        node = random.randrange(0, len(instance))
        node_type = 'cla'
        keep_cla = {node}
        keep_lit = set()
        for _ in range(sub_node_cnt):
            if node_type == 'cla':
                neighbor = list(set(instance[node]) - keep_lit)
                if len(neighbor) == 0:
                    break
                node = random.choice(neighbor)
                keep_lit |= {node}
                node_type = 'lit'
            else:
                neighbor = list(set(lit_2_cla[node]) - keep_cla)

                node_i = random.randrange(0, len(neighbor) + 1)
                if node_i == len(neighbor) and -node not in keep_lit:  # walk to the negation of the current literal
                    node = -node
                    keep_lit |= {node}
                else:
                    if len(neighbor) == 0:
                        break
                    node_i = random.randrange(0, len(neighbor))

                    node = neighbor[node_i]
                    keep_cla |= {node}
                    node_type = 'cla'

        new_instance = []
        for cla_i in keep_cla:
            cla = instance[cla_i]
            new_cla = list(set(cla) & keep_lit)
            new_instance += [new_cla]

        # A workaround for when instance becomes empty
        if len(new_instance) == 0:
            return [[1]]

        return remove_space(new_instance)

    def gcl_subgraph(self, instance, args):
        cla_cnt = len(instance)
        var_cnt = max([max(np.abs(x)) for x in instance])
        sub_node_cnt = int(np.ceil((cla_cnt + var_cnt) * (1 - args.gcl_subgraph_ratio)))

        lit_2_cla = {lit: [] for lit in list(range(1, var_cnt+1)) + list(range(-var_cnt, 0))}
        for cla_i, cla in enumerate(instance):
            for lit in cla:
                lit_2_cla[lit] += [cla_i]

        node = random.randrange(0, len(instance))
        node_type = 'cla'
        lit_neighbor = set()
        cla_neighbor = set()
        keep_cla = {node}
        keep_lit = set()
        for i in range(sub_node_cnt):
            if node_type == 'cla':
                lit_neighbor = list((set(lit_neighbor) | set(instance[node])) - keep_lit)
                if len(lit_neighbor) == 0:
                    break
                node = random.choice(lit_neighbor)
                keep_lit |= {node}
                node_type = 'lit'
            else:
                cla_neighbor = list((set(cla_neighbor) | set(lit_2_cla[node])) - keep_cla)

                node_i = random.randrange(0, len(cla_neighbor) + len(lit_neighbor) + 1)
                if node_i == len(cla_neighbor) + len(lit_neighbor) and -node not in keep_lit:  # walk to the negation of the current literal
                    node = -node
                    keep_lit |= {node}
                elif len(cla_neighbor) <= node_i:
                    node = lit_neighbor[node_i - len(cla_neighbor) - 1]
                    keep_lit |= {node}
                else:
                    if len(cla_neighbor) == 0:
                        break

                    node = cla_neighbor[node_i]
                    keep_cla |= {node}
                    node_type = 'cla'

        new_instance = []
        for cla_i in keep_cla:
            cla = instance[cla_i]
            new_cla = list(set(cla) & keep_lit)
            new_instance += [new_cla]

        # A workaround for when instance becomes empty
        if len(new_instance) == 0:
            return [[1]]

        return remove_space(new_instance)

    def augment_instance(self, args):
        #print("prev length " + str(len(self.instance)))
        new_instance = copy.deepcopy(self.instance)

        if args.reverse_aug:    
            #if random.uniform(0, 1) < self.sub:
            #    new_instance = self.subsume_clause_elimination(new_instance)
            if random.uniform(0, 1) < self.at:
                new_instance = self.add_trivial(new_instance, added_literal=args.at_added_literal, added_clause=args.at_added_clause)

            if random.uniform(0, 1) < self.ve:
                #num_variable = get_num_literal(self.instance)
                #if not args.study_ve:
                num_var = get_num_literal(new_instance)
                for _ in range(max(int(args.num_ve * num_var), 1)):
                    new_instance = self.variable_elimination(new_instance, eliminate_var=args.ve_eliminate_var, max_resolvent=args.ve_max_resolvent, ve_small=args.ve_small)
                #else:
                #    new_instance = self.variable_elimination(new_instance, eliminate_var=args.ve_eliminate_var, max_resolvent=args.ve_max_resolvent, pos_variable=args.ve_pos_variable)

                #new_instance = self.variable_elimination(new_instance, eliminate_var=args.ve_eliminate_var, max_resolvent=args.ve_max_resolvent)
                #new_instance = self.variable_elimination(new_instance, eliminate_var=args.ve_eliminate_var, max_resolvent=args.ve_max_resolvent)

                #print(" ")
            if random.uniform(0, 1) < self.cr:
                new_instance = self.clause_resolution(new_instance, added_prop=args.cr_added_resolv)


            if random.uniform(0, 1) < self.sub:
                new_instance = self.subsume_clause_elimination(new_instance)
			
            if random.uniform(0, 1) < self.gcl_cla:
                new_instance = self.gcl_cla_drop(new_instance, args)

            if random.uniform(0, 1) < self.gcl_var:
                new_instance = self.gcl_var_drop(new_instance, args)

            if random.uniform(0, 1) < self.gcl_link:
                new_instance = self.gcl_link_purt(new_instance, args)

            if random.uniform(0, 1) < self.gcl_sub:
                if args.DFS:
                    new_instance = self.gcl_subgraph_D(new_instance, args)
                else:
                    new_instance = self.gcl_subgraph(new_instance, args)
        else:
            if random.uniform(0, 1) < self.gcl_sub:
                if args.DFS:
                    new_instance = self.gcl_subgraph_D(new_instance, args)
                else:
                    new_instance = self.gcl_subgraph(new_instance, args)

            if random.uniform(0, 1) < self.gcl_link:
                new_instance = self.gcl_link_purt(new_instance, args)

            if random.uniform(0, 1) < self.gcl_var:
                new_instance = self.gcl_var_drop(new_instance, args)

            if random.uniform(0, 1) < self.gcl_cla:
                new_instance = self.gcl_cla_drop(new_instance, args)
            if random.uniform(0, 1) < self.sub:
                new_instance = self.subsume_clause_elimination(new_instance)

            if random.uniform(0, 1) < self.cr:
                new_instance = self.clause_resolution(new_instance, added_prop=args.cr_added_resolv)


            if random.uniform(0, 1) < self.ve:
                num_var = get_num_literal(new_instance)
                for _ in range(max(int(args.num_ve * num_var), 1)):
                    new_instance = self.variable_elimination(new_instance, eliminate_var=args.ve_eliminate_var, max_resolvent=args.ve_max_resolvent, ve_small=args.ve_small)

            if random.uniform(0, 1) < self.at:
                new_instance = self.add_trivial(new_instance, added_literal=args.at_added_literal, added_clause=args.at_added_clause)
            #if random.uniform(0, 1) < self.sub:
            #    new_instance = self.subsume_clause_elimination(new_instance)

        """ 
        print(len(new_instance))
        total_subsume = 0
        for i in range(len(new_instance)):
            for j in range(len(new_instance)):
                if i != j:
                    if set(new_instance[i]).issubset(set(new_instance[j])):
                        total_subsume += 1
        print(total_subsume)
        print(" ")
        """

        #print("current length " + str(len(new_instance)))

        return new_instance, get_num_literal(new_instance)


    def augment_two_instances(self, args):
        #augment1 = copy.deepcopy(self.instance)
        #augment2 = copy.deepcopy(self.instance)

        #while len(augment1) == len(augment2):
        #print(len(self.instance))
        #print(len(augment1))
        #print(" ")
        #augment1 = self.instance
        #augment2 = self.instance
        #while len(augment1) == len(augment2):
        augment1, n_var1 = self.augment_instance(args)
        augment2, n_var2 = self.augment_instance(args)
        #print(len(augment1), len(augment2))
        #print(len(augment1))
        #print(len(augment2))
        #print("")
        return augment1, augment2, n_var1, n_var2




