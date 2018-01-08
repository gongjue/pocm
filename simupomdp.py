#!/usr/bin/python

import os
import sys
import glob
import copy
import numpy as np

import hmm
import cm
import pocm
import util

from send_message import send_message

SOLVER_DIR = "/Users/gongjue/Downloads/pomdp-solve-5.4/"
MODEL_DIR = "/Users/gongjue/GDrive/Research/depression/pomdp_models/"

# def no_zero_matrix(M):
#     n = M.shape[0]
#     for i in range(n):
#         if M[i].sum == 0:
#             M[i,:] = 0
#     return

############ Depression Control ############

def reward(s, a, utility, cost):
    """
    Reward as a function of the true state and the price of action
    """
    # return [1,0.6,0.3,0][s]
    # return [1,0.6,0.3,0][s] - [0.05, 0.5][a]
    return utility[s] - cost[a]


def random_policy(p=0.3):
    """
    Random policy: choose I with probability of 0.3
    """
    return np.random.binomial(1, p, 1)[0]

def H_policy(b, threshold):
    """
    Threshold type of policy: if the belief on health (H) is lower than threshold,
    then the action is I, otherwise choose M.
    """
    if b[0] < threshold: 
        return 1
    else: 
        return 0

def S_policy(b, threshold):
    """
    Threshold type of policy: if the belief on severe depression (S) is higher 
    than threshold, then the action is I, otherwise choose M.
    """
    if b[2] > threshold: 
        return 1
    else: 
        return 0

def pomdp_policy(model_file, belief, horizon=None, discount=None):
    os.chdir(MODEL_DIR)
    horizon_option = ''
    if horizon is not None and horizon > 0:
        horizon_option = ' -horizon %d' %(horizon)
    else:
        horizon_option = ' -time_limit 2' 
    discount_option = ''
    if discount < 1 and discount > 0:
        discount_option = ' -discount %.4f' %(discount)
    ############# RUN ######### 
    os.system(SOLVER_DIR + 'src/pomdp-solve -pomdp ' + 
        MODEL_DIR + model_file + horizon_option +
        ' -stdout ' + MODEL_DIR + 'out.log')
    # Find the new generated file
    newest = max(glob.iglob('*.alpha'), key=os.path.getctime)
    # print newest
    # parse solution
    actions, alphas, next_nodes = parse_alpha(newest[:-6])
    # Find optimal action
    # print(alphas)
    values = [np.dot(belief, alpha) for alpha in alphas]
    ####### Clean new files ######### 
    os.remove(newest[:-5]+'pg')
    os.remove(newest)
    os.chdir('..')
    #### Return according to finite / infinite horizon
    if horizon:
        return np.argmax(values), actions, alphas, next_nodes
    else:
        return actions[np.argmax(values)]

def parse_alpha(fname):
    """Helper function
    """
    actions = []
    alphas = []
    next_nodes = []
    with open(fname+'.alpha') as f:
        lines = f.readlines()
    for i in range(len(lines)//3):
        actions.append( int(lines[3*i]) )
        alphas.append([float(a) for a in lines[3*i+1].split()]) 
    with open(fname+'.pg') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        next_node = lines[i].split()[2:]
        next_node_ = [i if n == '-' else int(n) for n in next_node]
        next_nodes.append( next_node_ )
    return actions, alphas, next_nodes



def expandZ(Z):
    # Add the Death state to emmision matrix
    ZZ = np.zeros((4,4))
    ZZ[:3,:3] = Z
    ZZ[3,3] = 1
    return ZZ


def update_model(fname, T=None, Z=None, utility=None, cost=None):
    # print "updating model..."
    fin = open(fname, "r")
    lines = fin.readlines()
    # for i, l in enumerate(lines):
    #     print i, l
    fin.close()
    if T is not None:
        lines[10] = "%f %f %f\n" %(T[0,0,0], T[0,0,1], T[0,0,2])
        lines[11] = "%f %f %f\n" %(T[0,1,0], T[0,1,1], T[0,1,2])
        lines[12] = "%f %f %f\n" %(T[0,2,0], T[0,2,1], T[0,2,2])
        lines[15] = "%f %f %f\n" %(T[1,0,0], T[1,0,1], T[1,0,2])
        lines[16] = "%f %f %f\n" %(T[1,1,0], T[1,1,1], T[1,1,2])
        lines[17] = "%f %f %f\n" %(T[1,2,0], T[1,2,1], T[1,2,2])
    if Z is not None:
        lines[20] = "%f %f %f\n" %(Z[0,0], Z[0,1], Z[0,2])
        lines[21] = "%f %f %f\n" %(Z[1,0], Z[1,1], Z[1,2])
        lines[22] = "%f %f %f\n" %(Z[2,0], Z[2,1], Z[2,2])
    if utility is not None:
        lines[24] = "R:Mon : * : H : * %.3f\n" % (utility[0] - cost[0])
        lines[25] = "R:Mon : * : M : * %.3f\n" % (utility[1] - cost[0])
        lines[26] = "R:Mon : * : S : * %.3f\n" % (utility[2] - cost[0])
        lines[27] = "R:Int : * : H : * %.3f\n" % (utility[0] - cost[1])
        lines[28] = "R:Int : * : M : * %.3f\n" % (utility[1] - cost[1])
        lines[29] = "R:Int : * : S : * %.3f\n" % (utility[2] - cost[1])
    fout = open(fname, "w")
    fout.writelines(lines)
    fout.close()

def simulate(membership, TM, TM0, TI0, Z, T1, s0, rho, _actions=None):
    # natural (true) transition
    TMn = membership[0,0]*TM[0] + membership[0,1]*TM[1] + membership[0,2]*TM[2]
    TIn = util.interaction_effect(TMn, rho)

    # t=1...T1
    actions = []
    observations = []
    s = s0
    # warm up loop
    if T1 > 0:
        for t in range(T1):
            if _actions is not None:
                a = _actions[t]
            else:
                a = np.random.binomial(1, 0.3, 1)[0]
            actions.append(a)
            # print s0, TMn[s0]
            s = np.random.choice(3, 1, p=TMn[s])[0]  # assumes 3 states
            # print expandZ(Z[s],s)
            o = np.random.choice(3, 1, p=Z[s])[0]
            observations.append(o)
            # print t,a,s,o
        hmm = hmm.HMM()
        hmm.pi = np.array([0.5, 0.3, 0.2])  # ASSUMPTION
        hmm.A = np.array([TM0,TI0])
        hmm.B = np.copy(Z)
        hmm.train(observations, actions, 0.01)
        T_hat = hmm.A
        Z_hat = hmm.B
    else:
        o = np.random.choice(3, 1, p=Z[s])[0]
        T_hat = np.array([TMn,TIn])
        Z_hat = Z
    b = Z[:,o]
    b = b / b.sum()  # initialize belief
    # personalize T, Z
    return TMn,TIn,actions,observations,s,b,T_hat,Z_hat

def solve(TMn,TIn,Z,actions,observations,
    s,b,T_hat,Z_hat,TM0,TI0,T2,
    utility, cost, discount,if_update,
    policy_type,threshold,model_file):
    # t = T1+1...T2
    # print Z_hat
    # hmm_ = HMM()
    # hmm_.pi = np.array([0.5, 0.3, 0.2])  # ASSUMPTION
    # hmm_.A = np.array([T_hat])
    # hmm_.B = np.copy(Z_hat)
    # hmm_ = copy.copy(hmm)
    actions = copy.copy(actions)
    observations = copy.copy(observations)

    err = 0
    rwd = 0
    A = []  # record the action of all periods
    B = []  # record the belief of all periods
    O = []  # record the observation of all periods
    S = []  # record the true state of all periods
    for t in range(T2):
        if policy_type == 0:
            a = random_policy()
        elif policy_type == 1:
            if t == 0:
                update_model(MODEL_DIR + model_file, T_hat, Z_hat, utility, cost)
                node, pomdp_actions, pomdp_alphas, pomdp_next_nodes = pomdp_policy(
                    model_file, b, horizon=T2, discount=discount)
                a = pomdp_actions[node]
                # for n in range(len(pomdp_actions)):
                #     print pomdp_actions[n], pomdp_alphas[n], pomdp_next_nodes[n]
                # print '%2d %2d' %(t, a)
            else:
                # print '%2d %2d %2d' %(t, a, o)
                node = pomdp_next_nodes[node][o]
                a = pomdp_actions[node]
        elif policy_type == 2:
            if t == 0:
                update_model(MODEL_DIR + model_file, T_hat, Z_hat, utility, cost)
            else:
                update_model(MODEL_DIR + model_file, T_hat, Z_hat)
            a = pomdp_policy(model_file, b, horizon=None,discount=discount)
            # print t, a
        elif policy_type == 3:
            a = H_policy(b,threshold)
        elif policy_type == 4:
            a = S_policy(b,threshold)
        actions.append(a)
        s = np.random.choice(3, 1, p=[TMn,TIn][a][s])[0]
        o = np.random.choice(3, 1, p=Z[s])[0]
        observations.append(o)
        _b = Z_hat[:,o] * b.dot(T_hat[a])  # update belief
        if _b.sum() > 0:
            b = _b / _b.sum()
        else:
            b = np.array([1./3.,1./3.,1./3.])
        # print np.array_str(Z_hat, precision=4, suppress_small=True)
        # print np.array_str(b, precision=4, suppress_small=True)

        if if_update > 0:
            hmm = HMM()
            hmm.pi = np.array([0.5, 0.3, 0.2])  # ASSUMPTION
            hmm.A = np.copy(T_hat)
            hmm.B = np.copy(Z)
            hmm.train(observations, actions, 0.01)
            T_hat = hmm.A
            Z_hat = hmm.B

        # print t,a,s,o
        rwd += reward(s, a, utility, cost) * np.power(discount,t)
        err += (1 - b[s])**2
        A.append(a)
        S.append(s)
        O.append(o)
        B.append(b)
    return 1. * err / T2, 1. * rwd / T2, A, S, O, B

####### TEST #######
def test_pomdp(N=10, T1=100, T2=20, s0=0, if_update=0):
    import pickle

    def repeat_experience(N,ptype,u,c,threshold,verbose=1):
        """
        Perform the policy for one membership with mutiple times
        """
        if verbose: 
            print(policy_name(ptype,threshold))
        POLICY_RESULT = {
            'Action':      [],
            'State':       [],
            'Observation': [],
            'Belief':      [],
            'Error':       [],
            'Reward':      []
        }
        for j in range(N):
            e,r,A,S,O,B = solve(TMn,TIn,Z,actions,observations,
                                s,b,T_hat,Z_hat,TM0,TI0,T2,
                                utility_set[u], cost_set[c], 
                                discount,if_update,
                                ptype,threshold,'depression.POMDP')
            POLICY_RESULT['Action'].append(A)
            POLICY_RESULT['State'].append(S)
            POLICY_RESULT['Observation'].append(O)
            POLICY_RESULT['Belief'].append(B)
            POLICY_RESULT['Error'].append(e)
            POLICY_RESULT['Reward'].append(r)
            if verbose > 0 and (j+1) % verbose == 0:
                print('%3d' %(j+1), end=' ')
                if (j+1) % (20*verbose) == 0: print('')
        if verbose > 0: print('')
        return POLICY_RESULT

    def policy_name(ptype,threshold):
        policy_types = ['random', 
                        'pomdp no update', 
                        'pomdp with updates', 
                        'S policy', 
                        'H_policy']
        if ptype < 3:
            return policy_types[ptype]
        else:
            return '%s: threshold: %s' %(policy_types[ptype], threshold)

    # a = pomdp_policy('depression.POMDP', [0.2, 0.5, 0.3])
    # print a
    TM = np.array([
        [
        [0.20,0.70,0.10],
        [0.00,0.60,0.40],
        [0.00,0.02,0.98]],
        [
        [0.56,0.44,0.00],
        [0.12,0.72,0.16],
        [0.00,0.27,0.73]],
        [
        [0.98,0.02,0.00],
        [0.02,0.97,0.01],
        [0.00,0.30,0.70]]])
    TM0 = np.array([
        [0.56, 0.42, 0.02],
        [0.20, 0.65, 0.15],
        [0.17, 0.50, 0.33]]) # 3x3
    TI0 = np.array([
        [0.95,0.04,0.01],
        [0.55,0.44,0.01],
        [0.30,0.60,0.10]]) # 3x3
    Z = np.array([
        [ 0.984391, 0.015000, 0.000609],
        [ 0.016382, 0.956973, 0.026645],
        [ 0.000635, 0.000058, 0.999307]]) # 3x3
    # update_model(MODEL_DIR+'depression.POMDP', np.array([TM0,TI0]), Z)
    # s0 = 0
    rho_set = [0.8, 0.2]
    # N = 10
    # T1 = 100
    # T2 = 20
    discount = 0.99
    # m = np.array([0.25,0.5,0.25]).reshape((1,3))
    membership_set = np.array([
        [1.,0.,0.],        # 0
        [0.,1.,0.],        # 1
        [0.,0.,1.],        # 2
        [1./3.,2./3.,0.],  # 3
        [1./3.,0.,2./3.],  # 4
        [0.,1./3.,2./3.],  # 5
        [0.,2./3.,1./3.],  # 6
        [2./3.,0.,1./3.],  # 7
        [2./3.,1./3.,0.],  # 8
        [1./3.,1./3.,1./3.],  # 9
        [0.25,0.5,0.25]    # 10
    ])


    utility_set = np.array([
        [1,0.4,0.2,0],
        [1,0.6,0.3,0],
        [1,0.8,0.5,0]
    ])
    cost_set = np.array([
        [0.05, 0.20],
        [0.01, 0.20],
        [0.01, 0.10]
    ])

    policy_types_in_use = [0,1,2,3,  3,  3,  3,  3,  4,  4,  4,  4,  4]
    thresholds_in_use =   [0,0,0,0.1,0.2,0.3,0.4,0.5,0.9,0.8,0.7,0.6,0.5]

    SIMU_RESULTS = {}
    # m = get_membership(5, [1.0,0.,0.])
    # m = np.array([0,0,1]).reshape((1,3))

    for mm in [1]: #range(3):
        for _rho in [1]: #range(len(rho_set)):
            m = membership_set[mm].reshape((1,3))
            rho = rho_set[_rho]
            print('m = ', m[0])
            TMn,TIn,actions,observations,s,b,T_hat,Z_hat = simulate(
                m, TM, TM0, TI0, Z, T1, s0, rho)
            # print (TMn)
            # print (TIn)
            # print (actions)
            # print (observations)
            # print (s)
            # print (b)
            # print (T_hat)
            # print (Z_hat)
            SIMU_RESULTS['TMn'] = TMn
            SIMU_RESULTS['TIn'] = TIn
            SIMU_RESULTS['actions'] = actions
            SIMU_RESULTS['observations'] = observations
            SIMU_RESULTS['T_hat'] = T_hat
            SIMU_RESULTS['Z_hat'] = Z_hat

            simu_file_name = "simu.%d.%d.%d.%d.p" %(s0,T1,mm,_rho)
            pickle.dump(SIMU_RESULTS, open( 
                "/Users/gongjue/GDrive/Research/depression/policy_out/"+simu_file_name, 
                "wb" ) )

            T_hat = np.array([TMn, TIn])
            Z_hat = Z

            for c in [2]: #range(len(cost_set)):
                for u in [0]: #range(len(utility_set)):
                    RANDOM_RESULT  = repeat_experience(N,policy_types_in_use[0],u,c,
                        thresholds_in_use[0],verbose=100)
                    POMDP_0_RESULT = repeat_experience(N,policy_types_in_use[1],u,c,
                        thresholds_in_use[1],verbose=100)
                    POMDP_1_RESULT = repeat_experience(N,policy_types_in_use[2],u,c,
                        thresholds_in_use[1],verbose=10)
                    S_1_RESULT     = repeat_experience(N,policy_types_in_use[3],u,c,
                        thresholds_in_use[3],verbose=100)
                    S_2_RESULT     = repeat_experience(N,policy_types_in_use[4],u,c,
                        thresholds_in_use[4],verbose=100)
                    S_3_RESULT     = repeat_experience(N,policy_types_in_use[5],u,c,
                        thresholds_in_use[5],verbose=100)
                    S_4_RESULT     = repeat_experience(N,policy_types_in_use[6],u,c,
                        thresholds_in_use[6],verbose=100)
                    S_5_RESULT     = repeat_experience(N,policy_types_in_use[7],u,c,
                        thresholds_in_use[7],verbose=100)
                    H_9_RESULT     = repeat_experience(N,policy_types_in_use[8],u,c,
                        thresholds_in_use[8],verbose=100)
                    H_8_RESULT     = repeat_experience(N,policy_types_in_use[9],u,c,
                        thresholds_in_use[9],verbose=100)
                    H_7_RESULT     = repeat_experience(N,policy_types_in_use[10],u,c,
                        thresholds_in_use[10],verbose=100)
                    H_6_RESULT     = repeat_experience(N,policy_types_in_use[11],u,c,
                        thresholds_in_use[11],verbose=100)
                    H_5_RESULT     = repeat_experience(N,policy_types_in_use[12],u,c,
                        thresholds_in_use[12],verbose=100)

                    EXP_RESULT = {}
                    EXP_RESULT['N'] = N
                    EXP_RESULT['m'] = m
                    EXP_RESULT['s0'] = s0
                    EXP_RESULT['T1'] = T1
                    EXP_RESULT['T2'] = T2
                    EXP_RESULT['rho'] = rho
                    EXP_RESULT['simu'] = SIMU_RESULTS
                    EXP_RESULT['random'] = RANDOM_RESULT
                    EXP_RESULT['pomdp0'] = POMDP_0_RESULT
                    EXP_RESULT['pomdp1'] = POMDP_1_RESULT
                    EXP_RESULT['s1'] = S_1_RESULT
                    EXP_RESULT['s2'] = S_2_RESULT
                    EXP_RESULT['s3'] = S_3_RESULT
                    EXP_RESULT['s4'] = S_4_RESULT
                    EXP_RESULT['s5'] = S_5_RESULT
                    EXP_RESULT['h9'] = H_9_RESULT
                    EXP_RESULT['h8'] = H_8_RESULT
                    EXP_RESULT['h7'] = H_7_RESULT
                    EXP_RESULT['h6'] = H_6_RESULT
                    EXP_RESULT['h5'] = H_5_RESULT
                    out_file_name = "exp.%d.%d.%d.%d.%d.%d.%d.p" %(s0, T1, T2,mm,_rho,u,c)
                    pickle.dump(EXP_RESULT, 
                        open( "/Users/gongjue/GDrive/Research/depression/policy_out/"+out_file_name, "wb" ) )
                    print('write results to ' + out_file_name)
                    # print cost, np.mean(POMDP_0_RESULT['Action'])
                    # os.system("terminal-notifier -message Done -sender com.apple.Safari")
    return


if __name__ == "__main__":
    # args = map(int, sys.argv[1:])
    # print('N=args[0], T1=args[1], s0=args[2], if_update=args[3]' %(args
    # test_pomdp(N=args[0], T1=args[1], s0=args[2], if_update=args[3])
    # send_message("Done: s0=%s, update=%d" %(args[2], args[3]))
    test_pomdp(N=1, T1=-1, s0=0, if_update=0)

