import json
from re import L
from typing import Counter 
import numpy as np
import math as m
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn


from itertools import compress
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

#### Functions ####
def RR_maker(event,neuron, pre_time, post_time, bin_size):
    binned_spikes = {}
    event_window = list(np.arange(pre_time, post_time, bin_size))
    total_bins = len(event_window)
    for event_label, event_ts in event.items():
        binned_spikes[event_label] = {}
        for neuron_label, neuron_ts in neuron.items():
            bin_spikes = np.empty((len(event_ts),total_bins))
            bin_spikes[:] = np.NaN
            for trial_i in range(len(event_ts)):
                spikes = [nst - event_ts[trial_i] for nst in neuron_ts]
                rel_spikes = np.array(spikes)
                # relative_spikes is the offset spike times for a given trial
                # np.histogram returns an array [histogram, bin_edges] so the call below only grabs the histogram
                bin_spikes[trial_i]  = np.histogram(rel_spikes, total_bins, range = (pre_time, post_time))[0]
            binned_spikes[event_label][neuron_label] = {}
            binned_spikes[event_label][neuron_label] = bin_spikes

    return binned_spikes

def gen_inputmaker(input, type, comp_want):
    out = {}
    if type == 'PCA':
        event_labels = input[0]
        TrialBin = input[1]
        pca_score = input[2]
        max = 0
        foddiler = {}
        for TB in TrialBin:
            splite = TB.split('B')
            test_max = int(splite[1])
        
            if test_max > max:
                max = test_max
        bin_size = max
        for row in range(len(event_labels)):            
            if event_labels[row] not in foddiler:
                foddiler[event_labels[row]] = {}
        event_trialbins = {}
        for event_label in foddiler.keys():
            event_trialbins[event_label] = []
            event_trialbins[event_label] = (event_labels.count(event_label))
        
        holder = np.array([])
        for event_label in foddiler.keys():
            for pca_i in range(len(pca_score.T)):
                foddiler[event_label][pca_i] = np.array([])
                for tb_i in range(event_trialbins[event_label]):
                    holder = np.append(holder, pca_score[tb_i][pca_i])
                    if len(holder) == bin_size:
                        foddiler[event_label][pca_i] = np.vstack([foddiler[event_label][pca_i],holder]) if foddiler[event_label][pca_i].size else holder
                        holder = np.array([])
            pca_score = pca_score[event_trialbins[event_label]:len(pca_score)]           #Zeroing out fodder 
        culled_pca_dict = {}
        for event_label, pca_fodder in foddiler.items():
            culled_pca_dict[event_label] = {}
            for pca_comp in pca_fodder.keys():
                if int(pca_comp) < comp_want:
                    culled_pca_dict[event_label][pca_comp] = []
                    culled_pca_dict[event_label][pca_comp] = pca_fodder[pca_comp]
        for event_label, pca_label in culled_pca_dict.items():
            out[event_label] = np.array([])
            for pca_vals in pca_label.values():
                holder = []
                holder = pca_vals
                out[event_label] = np.hstack([out[event_label],holder]) if out[event_label].size else holder
        return out, culled_pca_dict
    elif type == 'RR':
        for event_label, neuron_label in input.items():
            out[event_label] = np.array([])
            for neuron_ts in neuron_label.values():
                holder = []
                holder = neuron_ts
                out[event_label] = np.hstack([out[event_label],holder]) if out[event_label].size else holder
        return out
    else:
        print('{} format type has not been made yet'.format(type))
        
def PCA_inputmaker(RR):  
    true_labels = []
    TrialBin_labels = []
    PCA_input = np.array([])
    for event_label, neurons in RR.items():
        PCA_hold = np.array([])
        counter = 0
        count = 0
        for neuron_c in neurons.values():
            TL_hold = []
            TB_hold = []
            Val_hold = np.array([])
            stopper = len(neuron_c)*len(neuron_c[0])
            for trial_i in range(len(neuron_c)):
                for bin_i in range(len(neuron_c[trial_i])):
                    key = 'T{}B{}'.format(trial_i+1,bin_i+1)
                    if counter < stopper:
                        TB_hold.append(key)
                        TL_hold.append(event_label)
                        counter += 1
                    holder = neuron_c[trial_i][bin_i]
                    Val_hold = np.vstack([Val_hold,holder]) if Val_hold.size else holder
            PCA_hold = np.hstack([PCA_hold,Val_hold]) if PCA_hold.size else Val_hold
            if bool(TL_hold) == True:
                true_labels = true_labels + TL_hold
            if bool(TB_hold) == True:
                    TrialBin_labels = TrialBin_labels + TB_hold
        PCA_input = np.vstack([PCA_input,PCA_hold]) if PCA_input.size else PCA_hold
    out = [true_labels, TrialBin_labels, PCA_input]
    return out, true_labels, PCA_input

def LDA_LeaveOneOut(input):      # LDA Classification with Leave One Out
    #formatting for LDA input (BIG Nest List)
    PC_LDA_uncut = []
    for event_label, trial_ts in input.items():
        event_holder = []
        for trial_i in range(len(trial_ts)):
            event_holder.append(event_label)
        if not PC_LDA_uncut:
            PC_LDA_uncut = [event_holder, trial_ts]
        else:
            PC_LDA_uncut[0] = PC_LDA_uncut[0] + event_holder
            PC_LDA_uncut[1] = np.vstack([PC_LDA_uncut[1],trial_ts])
    
    LDA_predicted = []
    teachingset_events = PC_LDA_uncut[0]
    for test_i in range(len(PC_LDA_uncut[0])):
        test_trial = PC_LDA_uncut[1][test_i]
        teachingset_events = np.array(PC_LDA_uncut[0])
        teachingset_events = np.delete(teachingset_events,test_i)
        #del teachingset_events[test_i]
        teachingset_trials = np.delete(PC_LDA_uncut[1],test_i, axis=0)

        model = LinearDiscriminantAnalysis()
        model.fit(teachingset_trials,teachingset_events)
        pred = model.predict([test_trial])
        LDA_predicted.append(pred[0])

    # Confusion Matrix and Co
    LDA_result_confusion = confusion_matrix(PC_LDA_uncut[0], LDA_predicted)
    LDA_result_accuracy = accuracy_score(PC_LDA_uncut[0], LDA_predicted)

    #Hrow = sum((row_i_total/total_numbers)*log2(row_i_total/total_numbers))
    total_trials = len(LDA_predicted)
    H_row_temp = np.empty(len(LDA_result_confusion))
    for row_i in range(len(LDA_result_confusion)):
        p_row_i = sum(LDA_result_confusion[row_i])/total_trials
        H_row_temp[row_i] = p_row_i*m.log2(p_row_i)
    H_row = -sum(H_row_temp)

    #Hcol = sum((col_i_total/total_numbers)*log2(col_i_total/total_numbers))
    res_conf_T = LDA_result_confusion.T
    H_col_temp = np.empty(len(res_conf_T))
    for col_i in range(len(res_conf_T)):
        p_col_i = sum(res_conf_T[col_i])/total_trials
        H_col_temp[col_i] = p_col_i*m.log2(p_col_i)
    H_col = -sum(H_col_temp)

    #Helement = sum((element_i_val/total_numbers)*log2(element_i_val/total_numbers))
    p_ele_temp = np.empty(LDA_result_confusion.shape)
    for row_i in range(len(LDA_result_confusion)):
        for col_i in range(len(res_conf_T)):
            p_ele_i = LDA_result_confusion[row_i][col_i]/total_trials
            if p_ele_i == 0:
                p_ele_temp[row_i][col_i] = 0
            else:
                p_ele_temp[row_i][col_i] = p_ele_i*m.log2(p_ele_i)
    H_ele = -sum(sum(p_ele_temp))

    #Confusion Mutual Information = Hrow + Hcol - Helement
    LDA_confusion_mutual_info = H_row + H_col - H_ele

    return LDA_predicted, LDA_result_confusion, LDA_result_accuracy, LDA_confusion_mutual_info

def PSTH_LeaveOneOut(input):     # PSTH Classification with Leave One Out
    #### Euclidian Distances ####
    # Cut Trial N
    PCA_relative = {}
    test_trial = {}
    true_labels = []
    for event_label, trial_ts in input.items():
        PCA_relative[event_label] = {}
        test_trial[event_label] = np.array([])
        fodder = [None] * len(trial_ts)
        for trial_i in range(len(trial_ts)):
            PCA_relative[event_label][trial_i] = np.array([])
            test_trial[event_label] = np.vstack([test_trial[event_label], trial_ts[trial_i]]) if test_trial[event_label].size else trial_ts[trial_i]
            relative = np.delete(input[event_label],trial_i,0)
            # Beeg Array being made not being seperated by test trial like test trial array, may have to use dictionary entry per trial
            PCA_relative[event_label][trial_i] = np.vstack([PCA_relative[event_label][trial_i], relative]) if PCA_relative[event_label][trial_i].size else relative       
            fodder[trial_i] = event_label
        true_labels = true_labels + fodder          # NOTE Keep track of True Event

    # PSTH Relative Data (Make PSTH function see HW1) (RR w/o cut trial)
    relative_templates = {}
    for event_label, trials in PCA_relative.items():
        relative_templates[event_label] = {}
        for trial_i, arrays in trials.items():
            relative_templates[event_label][trial_i] = PSTH_builder(arrays)

    gen_templates = {}
    for event_label, trials in input.items():
        gen_templates[event_label] = {}
        for trial_i in range(len(trials)):
            gen_templates[event_label][trial_i] = PSTH_builder(trials)

    # Building the general template (Combination of Relative and original PSTH
    Template = {}
    for rel_label, rel_vals in relative_templates.items():
        Template[rel_label] = {}
        for gen_label, gen_vals in gen_templates.items():
            Template[rel_label][gen_label] = []
            if rel_label == gen_label:
                Template[rel_label][gen_label] = rel_vals
            else:
                Template[rel_label][gen_label] = gen_vals

    # Find Euclidian Distance
    E_dist = {}
    for rel_label, events in Template.items():
        E_dist[rel_label] = {}
        for event_label, trials in events.items():
            E_dist[rel_label][event_label] = {}
            for test_i in range(len(test_trial[rel_label])):
                cols_test = test_trial[rel_label][test_i]
                if rel_label == event_label:
                    cols_template = trials[test_i]
                else:
                    cols_template = trials[0]
                squares = (cols_template[:] - cols_test[:])**2
                E_dist[rel_label][event_label][test_i] = m.sqrt(sum(squares))

    # Sort into Predicted Event
    comp_temp = {}
    for rel_label, events in E_dist.items():
        comp_temp[rel_label] = np.array([])
        for trials in events.values():
            fod = np.zeros([len(trials)])
            for trial_i in range(len(trials)):
                fod[trial_i] = trials[trial_i]
            comp_temp[rel_label] = np.vstack([comp_temp[rel_label], fod]) if comp_temp[rel_label].size else fod

    labels = list(comp_temp.keys())
    pred_labels = []
    for rel_label, temp in comp_temp.items():
        for col in temp.T:
            min_value = np.amin(col)
            col_bool = np.less_equal(col,min_value)
            foddest = list(compress(labels, col_bool))
            pred_labels = pred_labels + foddest

    #### Confusion Matrix ####
    # true_events = list of true event labels in order of classification
    # predicted_events = list of predicted event labels in order of classification
    result_confusion = confusion_matrix(true_labels, pred_labels)
    result_accuracy = accuracy_score(true_labels, pred_labels)

    #### Entropy ####

    #Hrow = sum((row_i_total/total_numbers)*log2(row_i_total/total_numbers))
    total_trials = len(pred_labels)
    H_row_temp = np.empty(len(result_confusion))
    for row_i in range(len(result_confusion)):
        p_row_i = sum(result_confusion[row_i])/total_trials
        H_row_temp[row_i] = p_row_i*m.log2(p_row_i)
    H_row = -sum(H_row_temp)

    #Hcol = sum((col_i_total/total_numbers)*log2(col_i_total/total_numbers))
    res_conf_T = result_confusion.T
    H_col_temp = np.empty(len(res_conf_T))
    for col_i in range(len(res_conf_T)):
        p_col_i = sum(res_conf_T[col_i])/total_trials
        H_col_temp[col_i] = p_col_i*m.log2(p_col_i)
    H_col = -sum(H_col_temp)


    #Helement = sum((element_i_val/total_numbers)*log2(element_i_val/total_numbers))

    p_ele_temp = np.empty(result_confusion.shape)
    for row_i in range(len(result_confusion)):
        for col_i in range(len(res_conf_T)):
            p_ele_i = result_confusion[row_i][col_i]/total_trials
            if p_ele_i == 0:
                p_ele_temp[row_i][col_i] = 0
            else:
                p_ele_temp[row_i][col_i] = p_ele_i*m.log2(p_ele_i)
    H_ele = -sum(sum(p_ele_temp))

    #Confusion Mutual Information = Hrow + Hcol - Helement
    confusion_mutual_info = H_row + H_col - H_ele

    return pred_labels, result_confusion, result_accuracy, confusion_mutual_info

def PSTH_builder(RR):
    summer = np.sum(RR,axis=0)
    PSTH = summer/len(RR)
    return PSTH

#### Importing and Organizing Data ####
# Import Data
with open('.\HW4\hw4.json') as f:
    data = json.load(f)

pre_time = 0; post_time = .2; bin_size = .01;     #Actually 1ms not 100ms
window_len = list(np.arange(pre_time, post_time, bin_size))

# Put Data into Trials x (Neurons * Bins)
rr_matrix = RR_maker(data['events'],data['neurons'], pre_time, post_time, bin_size)     
PCA_input_full, true_labels, PCA_input = PCA_inputmaker(rr_matrix)

## Scatter plot of Raw Channel 1 ##

fig = plt.figure(figsize=(10,8))
ax = plt.axes()
plt.scatter(range(len(PCA_input.T[0])),PCA_input.T[0])
plt.title('Scatter Plot of Channel 1')
#plt.show()
plt.savefig('.\FinalProject\Charts\ScatPlot_Chan1.png')

#### Running PCA ####
components_wanted = 5

### Standardizing with Z score ###
stand_pca_input = np.array([])
bad_col = []
for col in range(len(PCA_input.T)):
    stand_pca_input_hold = stats.zscore(PCA_input.T[col], ddof=1)
    #if all(element == norm_pca_input_hold[0] for element in norm_pca_input_hold):
    if np.isnan(stand_pca_input_hold).any() == False:     
        stand_pca_input = np.vstack([stand_pca_input, stand_pca_input_hold])   if stand_pca_input.size else stand_pca_input_hold
    else: 
        bad_col = col
for event_labels, neurons in rr_matrix.items():
    for i in range(len(neurons.keys())):
        if i == bad_col:
            del neurons[list(neurons.keys())[i]]
            break
stand_pca_input = stand_pca_input.T

## Scatter Plot of Standardized Channel 1 ##

'''fig = plt.figure(figsize=(10,8))
ax = plt.axes()
plt.scatter(range(len(stand_pca_input.T[0])),stand_pca_input.T[0])
plt.title('Standardized Scatter Plot of Channel 1')
#plt.show()
plt.savefig('.\FinalProject\Charts\Stand_ScatPlot_Chan1.png') '''

pca_stand = PCA()
pca_stand.fit(stand_pca_input)
pca_stand_score = pca_stand.fit_transform(stand_pca_input)

stand_eigenvalues = pca_stand.explained_variance_.tolist()
stand_pc_variance = (pca_stand.explained_variance_ratio_ * 100).tolist()

## Reshaping PCA score back to T x (N*B) ##
PCA_stand_full = [PCA_input_full[0], PCA_input_full[1], pca_stand_score]
PCA_stand_long, culled_stand_pca = gen_inputmaker(PCA_stand_full,'PCA',components_wanted)     #PCA Long is the wanted output, culled_pca is for graphing

## PSTH of PC Channel 1 ##
view = list(np.arange(pre_time, post_time + .02, .02))
count = False
for event_label, pca_poopoo in culled_stand_pca.items():
    for pca_label, pca_vals in pca_poopoo.items():
        if not count:
            chan_psth = PSTH_builder(pca_vals)
            fig = plt.figure(figsize=(10,5))
            plt.bar(window_len,chan_psth, width = bin_size)

            plt.xticks(view)
            plt.xlabel('Time (s)')
            plt.title('Standardized {} PC # {} PSTH'.format(event_label,pca_label+1))
            plt.savefig('.\FinalProject\Charts\Standardized_{}_PC#{}PSTH.png'.format(event_label,pca_label+1))
            count = True

## PC LDA ##
LDA_stand_predicted, LDA_stand_result_confusion, LDA_stand_result_accuracy, LDA_stand_confusion_mutual_info = LDA_LeaveOneOut(PCA_stand_long)

## PC PSTH ##
PSTH_stand_predicted, PSTH_stand_result_confusion, PSTH_stand_result_accuracy, PSTH_stand_confusion_mutual_info = PSTH_LeaveOneOut(PCA_stand_long)

## Heat Maps of Confusion Matrices ##
#PSTH 
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(PSTH_stand_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(PSTH_stand_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('Standardized PSTH Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\FinalProject\Charts\stand_PSTH_Confusion_HeatMap.png')

#LDA
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(LDA_stand_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(LDA_stand_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('Standardized LDA Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\FinalProject\Charts\stand_LDA_Confusion_HeatMap.png')

### Regularizing with Vector Norms ###
L1_pca_input = np.array([])
L2_pca_input = np.array([])
Linf_pca_input = np.array([])
bad_col = []
for col in PCA_input.T:
    norm_L2 = np.linalg.norm(col)
    norm_Linf = np.linalg.norm(col, np.inf)

    L2_pca_input_hold = col/norm_L2 
    Linf_pca_input_hold = col/norm_Linf 
    #if all(element == norm_pca_input_hold[0] for element in norm_pca_input_hold):
    if np.isnan(L2_pca_input_hold).any() == False:     
        L2_pca_input = np.vstack([L2_pca_input, L2_pca_input_hold])   if L2_pca_input.size else L2_pca_input_hold
    else: 
        bad_col = col
    if np.isnan(Linf_pca_input_hold).any() == False:     
        Linf_pca_input = np.vstack([Linf_pca_input, Linf_pca_input_hold])   if Linf_pca_input.size else Linf_pca_input_hold
    else: 
        bad_col = col

L2_pca_input = L2_pca_input.T
Linf_pca_input = Linf_pca_input.T

'''## Scatter Plot of L2 Norm Channel 1 ##
fig = plt.figure(figsize=(10,8))
ax = plt.axes()
plt.scatter(range(len(L2_pca_input.T[0])),L2_pca_input.T[0])
plt.title('L2 Normalized Scatter Plot of Channel 1')
#plt.show()
plt.savefig('.\FinalProject\Charts\L2_Norm_ScatPlot_Chan1.png')

## Scatter Plot of L2 Norm Channel 1 ##
fig = plt.figure(figsize=(10,8))
ax = plt.axes()
plt.scatter(range(len(Linf_pca_input.T[0])),Linf_pca_input.T[0])
plt.title('Linf Normalized Scatter Plot of Channel 1')
#plt.show()
plt.savefig('.\FinalProject\Charts\Linf_Norm_ScatPlot_Chan1.png')'''


pca_L2 = PCA()
pca_Linf = PCA()
pca_L2.fit(L2_pca_input)
pca_Linf.fit(Linf_pca_input)
pca_L2_score = pca_L2.transform(L2_pca_input)
pca_Linf_score = pca_Linf.transform(Linf_pca_input)

L2_eigenvalues = pca_L2.explained_variance_.tolist()
Linf_eigenvalues = pca_Linf.explained_variance_.tolist()
L2_pc_variance = (pca_L2.explained_variance_ratio_ * 100).tolist()
Linf_pc_variance = (pca_Linf.explained_variance_ratio_ * 100).tolist()

#### Reshaping PCA score back to T x (N*B) ####

PCA_L2_full = [PCA_input_full[0], PCA_input_full[1], pca_L2_score]
PCA_Linf_full = [PCA_input_full[0], PCA_input_full[1], pca_Linf_score]
components_wanted = 5
PCA_L2_long, culled_L2_pca = gen_inputmaker(PCA_L2_full,'PCA',components_wanted)     #PCA Long is the wanted output, culled_pca is for graphing
PCA_Linf_long, culled_Linf_pca  = gen_inputmaker(PCA_Linf_full,'PCA',components_wanted)

## PSTH of PC Channel 1 ##

view = list(np.arange(pre_time, post_time + .02, .02))
count = False
for event_label, pca_poopoo in culled_L2_pca.items():
    for pca_label, pca_vals in pca_poopoo.items():
        if not count:
            chan_psth = PSTH_builder(pca_vals)
            fig = plt.figure(figsize=(10,5))
            plt.bar(window_len,chan_psth, width = bin_size)

            plt.xticks(view)
            plt.xlabel('Time (s)')
            plt.title('L2 Norm {} PC # {} PSTH'.format(event_label,pca_label+1))
            plt.savefig('.\FinalProject\Charts\L2_Normalized_{}_PC#{}PSTH.png'.format(event_label,pca_label+1))
            count = True

view = list(np.arange(pre_time, post_time + .02, .02))
count = False
for event_label, pca_poopoo in culled_Linf_pca.items():
    for pca_label, pca_vals in pca_poopoo.items():
        if not count:
            chan_psth = PSTH_builder(pca_vals)
            fig = plt.figure(figsize=(10,5))
            plt.bar(window_len,chan_psth, width = bin_size)

            plt.xticks(view)
            plt.xlabel('Time (s)')
            plt.title('Linf Norm {} PC # {} PSTH'.format(event_label,pca_label+1))
            plt.savefig('.\FinalProject\Charts\Linf_Normalized_{}_PC#{}PSTH.png'.format(event_label,pca_label+1))
            count = True


#### PC LDA ####
LDA_L2_predicted, LDA_L2_result_confusion, LDA_L2_result_accuracy, LDA_L2_confusion_mutual_info = LDA_LeaveOneOut(PCA_L2_long)
LDA_Linf_predicted, LDA_Linf_result_confusion, LDA_Linf_result_accuracy, LDA_Linf_confusion_mutual_info = LDA_LeaveOneOut(PCA_Linf_long)

#### PC PSTH ####
PSTH_L2_predicted, PSTH_L2_result_confusion, PSTH_L2_result_accuracy, PSTH_L2_confusion_mutual_info = PSTH_LeaveOneOut(PCA_L2_long)
PSTH_Linf_predicted, PSTH_Linf_result_confusion, PSTH_Linf_result_accuracy, PSTH_Linf_cnfusion_mutual_info = PSTH_LeaveOneOut(PCA_Linf_long)

## Heat Maps of Confusion Matrices ##
#PSTH 
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(PSTH_L2_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(PSTH_L2_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('L2 Norm PSTH Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\\FinalProject\\Charts\\L2Norm_PSTH_Confusion_HeatMap.png')

fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(PSTH_Linf_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(PSTH_Linf_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('Linf Norm PSTH Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\\FinalProject\\Charts\\LinfNorm_PSTH_Confusion_HeatMap.png')

#LDA
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(LDA_L2_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(LDA_L2_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('L2 Norm LDA Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\\FinalProject\\Charts\\L2Norm_LDA_Confusion_HeatMap.png')

fig = plt.figure(figsize=(12,12))
ax = plt.axes()
sn.set(font_scale=1.4) # for label size
sn.heatmap(LDA_Linf_result_confusion, annot=True, annot_kws={"size": 16}, linewidths=.5) # font size
plt.xlabel('True Events \n Peformance = {} %'.format(LDA_Linf_result_accuracy * 100))
plt.ylabel('Predicted Events')
plt.title('Linf Norm LDA Confusion Matrix')
ax.set_xticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 30)
ax.set_yticklabels(['CW Fast', 'CW Slow', 'CCW Fast', 'CCW Slow'], rotation = 45)
plt.savefig('.\\FinalProject\\Charts\\LinfNorm_LDA_Confusion_HeatMap.png')

#### Formatting Data For Output ####
results = {}

#PCA Input
'''results['pca_input'] = {}
for event_label in rr_matrix.keys():
    for neuron_label in rr_matrix[event_label].keys():            
        if neuron_label not in results['pca_input']:
            results['pca_input'][neuron_label] = {}
for i in range(len(norm_pca_input.T)):
    results['pca_input'][list(results['pca_input'].keys())[i]] = (np.unique(norm_pca_input.T[i])).tolist()

#Eigenvalues and PC_variance
results['eigenvalues'] = []
results['eigenvalues'] = eigenvalues
results['pc_variance'] = []
results['pc_variance'] = pc_variance

#PC PSTH
for event_label, pca_comps in PC_PSTH.items():
    results[event_label] = {}
    for pca_comp, pca_vals in pca_comps.items():
        results[event_label]['pc_{}_psth'.format(int(pca_comp)+1)] = []
        results[event_label]['pc_{}_psth'.format(int(pca_comp)+1)] = pca_vals.tolist()

#PSTH Classifier
results['psth_classifier'] = {}
results['psth_classifier']['performace'] = []
results['psth_classifier']['performace'] = result_accuracy.tolist()
results['psth_classifier']['confustion_matrix'] = []
results['psth_classifier']['confustion_matrix'] = result_confusion.tolist()
results['psth_classifier']['mutual_info'] = []
results['psth_classifier']['mutual_info'] = confusion_mutual_info.tolist()

#### Dumping to JSON ####
with open('.\HW4\Eppstein_Aaron_hw4_results.json', 'w') as f_out:
    json.dump(results, f_out,indent=4,sort_keys=True)'''
