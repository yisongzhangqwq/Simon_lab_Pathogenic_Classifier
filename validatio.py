#! / usr / bin / env python3
import numpy as np
import sys
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import auc, matthews_corrcoef
from scipy.stats import *
from sklearn.model_selection import train_test_split
import os
import glob

np.set_printoptions(precision=2)
#########################################################################
'''testing the threshold, at which the best MCC is obtained, using 20% of the data'''

# genes = ['g6pd','alas2','rs1','mtm1','otc','phex','f8','il2rg','l1cam','clcn5','ids','gla','abcd1','f9','gjb1','avpr2','pdha1','btk','ocrl','ndp','hprt1']
# genes = ['dcx', 'frmd7', 'nyx', 'cd40lg', 'rpgr', 'porcn', 'rp2', 'gpr143', 'atrx', 'atp7a', 'pcdh19']
current_file_path = os.getcwd()
print(f"Current directory is {current_file_path}")

genes_filename_list = glob.glob(f"{current_file_path}/merge_db/*_merged.tsv")  # find matched files
genes = [name.split("\\")[-1].split('_')[0] for name in genes_filename_list]  # use filenames to get genenames
genes.sort()
##############################################
genes_transcripts = pd.read_excel('genes_list.xlsx')  # where transcript IDs for transcript of interest os stored
vest4_mcc_identified_for_all_genes = []
revel_mcc_identified_for_all_genes = []
mean_vest4_threshold_identified_using_training_data = []
mean_revel_threshold_identified_using_training_data = []
stdev_revel_threshold_identified_using_training_data = []
stdev_vest4_threshold_identified_using_training_data = []
revel_mcc_mean_for_all_genes = []
vest4_mcc_mean_for_all_genes = []
revel_mcc_stdev_for_all_genes = []
vest4_mcc_stdev_for_all_genes = []
revel_precision_tested_mean_for_all_genes = []
vest4_precision_tested_mean_for_all_genes = []
revel_precision_tested_stdev_for_all_genes = []
vest4_precision_tested_stdev_for_all_genes = []
revel_recall_tested_mean_for_all_genes = []
vest4_recall_tested_mean_for_all_genes = []
revel_recall_tested_stdev_for_all_genes = []
vest4_recall_tested_stdev_for_all_genes = []
revel_accuracy_tested_mean_for_all_genes = []
vest4_accuracy_tested_mean_for_all_genes = []
revel_accuracy_tested_stdev_for_all_genes = []
vest4_accuracy_tested_stdev_for_all_genes = []
original_revel_precision_tested_mean_for_all_genes = []
original_revel_precision_tested_stdev_for_all_genes = []
original_revel_recall_tested_mean_for_all_genes = []
original_revel_recall_tested_stdev_for_all_genes = []
original_revel_accuracy_tested_mean_for_all_genes = []
original_revel_accuracy_tested_stdev_for_all_genes = []

bad_files_first = []
bad_files_second = []
results_file = open(current_file_path + '/results_2020_8_3.tsv', 'w')
results_file.write(f"Genes\tnumbers of gnomAD\tnumbers of HGMD\trevel default threshold\trevel MCC at default threshold\tvest4 "
                   f"default threshold\tvest4 MCC at default threshold\tGene-specific threshold\tOptimized "
                   f"MCC\tGene-specific "
                   f"threshold\tOptimized MCC\n")
mean_and_std_file_default = open(current_file_path + '/mean_and_std_default.tsv', 'w')
mean_and_std_file_optimized = open(current_file_path + '/mean_and_std_optimized.tsv', 'w')
# f'{gene}\t{mean_revel}\t{std_revel}\t{mean_vest4}\t{std_vest4}\n'
mean_and_std_file_default.write(f"gene\tnumbers of gnomAD\tnumbers of HGMD\tmean revel\tstd revel\tmean vest4\tstd vest4\n")
mean_and_std_file_optimized.write(f"gene\tnumbers of gnomAD\tnumbers of HGMD\tmean revel\tstd revel\tmean vest4\tstd vest4\n")
for gene in genes:
    print(f"Processing {gene}_merged.tsv")
    predictions_data = pd.read_csv(current_file_path + '/merge_db/' + gene + '_merged.tsv', sep='\t', index_col=None)
    file_to_write = open(current_file_path + '/results/' + gene + '_results.txt', 'w')

    predictions_data_noTestClass = predictions_data[
        predictions_data['class'] != 'test']  # exclude variants labelled as "test"
    vest4_thresholds_identified_using_training_data = []
    revel_thresholds_identified_using_training_data = []
    revel_mcc_at_identified_threshold = []  # 10 validated mcc scores collected
    vest4_mcc_at_identified_threshold = []  # 10 validated mcc scores collected
    revel_precision_tested_at_identified_threshold = []
    revel_recall_tested_at_identified_threshold = []
    revel_accuracy_tested_at_identified_threshold = []
    vest4_precision_tested_at_identified_threshold = []
    vest4_recall_tested_at_identified_threshold = []
    vest4_accuracy_tested_at_identified_threshold = []
    revel_precision_tested_at_original_threshold = []
    revel_recall_tested_at_original_threshold = []
    revel_accuracy_tested_at_original_threshold = []

    revel_mcc_l = []
    vest4_mcc_l = []

    count_occurance_result = predictions_data.class_binary.value_counts()

    try:
        occurance_0 = count_occurance_result[0]
    except KeyError:
        occurance_0 = 0
    try:
        occurance_1 = count_occurance_result[1]
    except KeyError:
        occurance_1 = 0

    transcript = genes_transcripts.query('gene==@gene')['transcript'].iloc[0].strip()  # obtaining the transcript used
    # print(f"transcript retrieved: '{transcript}'\n")

    predictions_data_noTestClass = predictions_data_noTestClass[
        predictions_data_noTestClass.Ensembl_transcriptid.str.contains(transcript)]
    predictions_data_noTestClass = predictions_data_noTestClass[predictions_data_noTestClass.REVEL_score != '.']

    #remove 0 when 1 exits
    pathogenic_positions = predictions_data_noTestClass[predictions_data_noTestClass.class_binary !=0]['pos(1-based)'].tolist()
    predictions_data_noTestClass = predictions_data_noTestClass.drop(predictions_data_noTestClass[(predictions_data_noTestClass['pos(1-based)'].isin(pathogenic_positions)) & (predictions_data_noTestClass['class_binary'] == 0)].index)
    count_jumped = 0
    if predictions_data_noTestClass.shape[0] == 0:
        print(f"{gene}_merged.tsv has 0 valid row after dropping '.'! Jumping this file!")
        bad_files_first.append(gene)
        count_jumped += 1
        file_to_write.close()
        continue
    for j in range(10):
        break_indicator = False
        random_seed = j
        '''separate a subset and form the test subset'''
        training_set, test_set, y_train, y_test = train_test_split(predictions_data_noTestClass,
                                                                   range(len(predictions_data_noTestClass)),
                                                                   test_size=0.25, random_state=j)
        '''a different approach to sampling data'''
        # test_set_df = predictions_data.sample(frac=0.3, random_state=random_seed) #sampling 25% of the data
        # test_set = test_set_df.to_csv('/Users/mdefsss2/x_linked_genes/'+gene+'/split_test_set.csv') # 'index=False' to exclude index column
        # training_set_df = predictions_data.loc[~predictions_data.index.isin(test_set_df.index)] #the remaining data, i.e. whatever is not in test set
        # training_set = training_set_df.to_csv('/Users/mdefsss2/x_linked_genes/'+gene+'/split_training_set.csv', sep='\t')
        file_to_write.write(f"\n*************************   Run: {str(random_seed)} {gene.upper()}\n")
        pathogenicity_threshold_numbers = 21
        pathogenicity_threshold = [i for i in np.linspace(0, 1, pathogenicity_threshold_numbers)]
        # data = pd.read_excel('/Users/mdefsss2/' + gene + '/' + gene + '_analysis.xlsx')
        # training_set = pd.read_csv('/Users/mdefsss2/x_linked_genes/' + gene + '/' + data_set + '_set.csv')
        var_class = training_set['class'].tolist()
        classes = training_set['class_binary'].tolist()
        ensembl_transcripts = training_set['Ensembl_transcriptid'].tolist()
        coordinates = training_set['pos(1-based)'].tolist()
        revel_scores = training_set['REVEL_score'].tolist()
        vest4_scores = training_set['VEST4_score'].tolist()
        list_index_to_drop = []

        transcript_found = True  # the predictions for the variant in transcript og interest not always available

        vest4_predictions = []
        count_error = 0
        for i in range(len(ensembl_transcripts)):
            strings = str(ensembl_transcripts[i])
            if_semicolon_in = False
            if ';' in strings:
                if_semicolon_in = True
                # print(f'Multiple transcripts found when processing {ensembl_transcripts[i]}')
                num_of_predictions = ensembl_transcripts[i].split(';')
                vest4_pred = vest4_scores[i].split(';')
                count_found_prediction = 0
                for k in range(len(num_of_predictions)):
                    if num_of_predictions[k] == transcript:
                        count_found_prediction += 1
                        try:
                            vest4_predictions.append(float(vest4_pred[k]))
                        except ValueError as e:
                            # transcript_found = False
                            # print(f"Error occurred while converting {vest4_pred[k]}\nDropping this row!")
                            # print(e)  # the transcript of interest wasn't available & the first transcript prediction
                            list_index_to_drop.append(i)
                            # is therefore chosen instead
                            # vest4_predictions.append(float(
                            #     vest4_pred[k + 1]))  # the next transcript is chosen but not considered for analysis

                    # elif transcript not in num_of_predictions:
                    # print('Transcript of interest MISSING!! at: ', coordinates[k])
                if count_found_prediction != 1:
                    count_error += 1
                    print(f'Multiple transcripts found when processing {ensembl_transcripts[i]}')
                    if count_found_prediction == 0:
                        print(
                            f"No matched transcript found!\nAn error occurred when processing {ensembl_transcripts[i]}\n")
                    else:
                        print(
                            f"{count_found_prediction} transcript found!\nAn error occurred when processing {ensembl_transcripts[i]}\n")
            elif strings.strip() == transcript.strip():
                vest4_predictions.append(float(vest4_scores[i]))
            else:
                count_error += 1
                print(f"Unexpected situation encountered!\nAn error occurred when processing {ensembl_transcripts[i]}")
                print(f"The string is '{strings}' while '{transcript.strip()}' is expected.")
                print(f"Semicolon : {if_semicolon_in}\n")
        if count_error:
            print(f'{count_error} error(s) encountered!\n')

        ########################################
        '''improving MCC scores for VEST4 in cases where there are multiple predictions for multiplr transcripts, where sorting by scores in EXCEL doesn't do the job.
        Here, the scores for transcript of interest are written to a new file along with the classes so they can be sorted to identify a better pathogenicity threshold'''
        # np.savetxt('/Users/mdefsss2/dbNSFP4/'+gene+'_vest_scores.csv', [p for p in zip(classes, vest4_predictions)], delimiter=',', fmt='%s')
        ########################################
        revel_predicted_pathogenic = []
        revel_predicted_nonpathogenic = []
        vest4_predicted_pathogenic = []
        vest4_predicted_nonpathogenic = []

        list_index_to_drop.sort(reverse=True)
        if list_index_to_drop:
            # print(f"Removing these index :{list_index_to_drop}")
            print(f"Going to remove {len(list_index_to_drop)} row(s)")
            for i in list_index_to_drop:
                del classes[i]
                del revel_scores[i]
            list_index_to_drop = []

        '''Mann Whitney U test'''
        if len(classes) == 0:
            if gene not in bad_files_second:
                bad_files_second.append(gene)
            print(f"{gene}_merged.tsv has 0 valid row!\nJumping this file!")
            break_indicator = True
            break
        if len(vest4_predictions) < len(classes):
            print(
                f"The length of vest4_predictions is smaller than classes!\nLengths for them is {len(vest4_predictions)} and {len(classes)} respectively.\nExiting!\n")
            sys.exit()
        for i in range(len(classes)):
            if classes[i] == 1:
                revel_predicted_pathogenic.append(revel_scores[i])
                vest4_predicted_pathogenic.append(vest4_predictions[i])
            elif classes[i] == 0:
                revel_predicted_nonpathogenic.append(revel_scores[i])
                vest4_predicted_nonpathogenic.append(vest4_predictions[i])
            else:
                print("Illegal classes indicator detected in Mann Whitney U test block!\n\n")
        # print('number of pathogenic vars: ', len(revel_predicted_pathogenic))
        # print('number of non-pathogenic vars: ', len(revel_predicted_nonpathogenic))
        # print('MannU test for REVEL: ', mannwhitneyu(revel_predicted_pathogenic, revel_predicted_nonpathogenic))
        # print('MannU test for VEST4: ', mannwhitneyu(vest4_predicted_pathogenic, vest4_predicted_nonpathogenic))
        #########################################################################
        '''for a number of pathogenicity thresholds, calculate the MCC scores for both tools'''
        binary_class = []
        for i in range(len(classes)):
            if classes[i] == 0 or classes[i] == 1:
                binary_class.append(classes[i])
            else:
                print("Illegle classes indicator detected in MCC block!\n\n")

        revel_MCC_at_diff_thresholds = []
        revel_max_mcc = 0
        revel_max_mcc_threshold = 0
        prec_revel = []
        rec_revel = []

        for k in range(len(pathogenicity_threshold)):
            revel_pred = []
            for i in range(len(revel_scores)):
                if float(revel_scores[i]) >= pathogenicity_threshold[k]:
                    i = 1
                    revel_pred.append(i)
                elif float(revel_scores[i]) < pathogenicity_threshold[k]:
                    i = 0
                    revel_pred.append(i)
            counter_tp = 0
            counter_fn = 0
            counter_fp = 0
            counter_tn = 0
            for m in range(len(revel_predicted_pathogenic)):
                if float(revel_predicted_pathogenic[m]) >= pathogenicity_threshold[k]:
                    counter_tp += 1
                elif float(revel_predicted_pathogenic[m]) < pathogenicity_threshold[k]:
                    counter_fn += 1
            for l in range(len(revel_predicted_nonpathogenic)):
                if float(revel_predicted_nonpathogenic[l]) >= pathogenicity_threshold[k]:
                    counter_fp += 1
                    # print('counter_fp: ', counter_fp)
                elif float(revel_predicted_nonpathogenic[l]) < pathogenicity_threshold[k]:
                    counter_tn += 1
            prec_revel.append(
                counter_tp / (counter_tp + counter_fp) if counter_tp + counter_fp != 0 and counter_tp != 0 else 0)
            rec_revel.append(counter_tp / (counter_tp + counter_fn) if counter_tp + counter_fn != 0 else 0)
            if counter_tp + counter_tn + counter_fp + counter_fn != 0:
                accuracy = (counter_tp + counter_tn) / (counter_tp + counter_tn + counter_fp + counter_fn)
            else:
                accuracy = 0
            if len(binary_class) != len(revel_pred):
                print(len(binary_class), len(revel_pred))
            revel_mcc = matthews_corrcoef(binary_class, revel_pred)
            if pathogenicity_threshold[k] == 0.5:  # suggested threshold
                # print('REVEL Precision @ ', pathogenicity_threshold[k], '=', prec_revel[k])
                # print('REVEL Recall @ ', pathogenicity_threshold[k], '=', rec_revel[k])
                # print(gene.upper(), ' REVEL MCC: ', revel_mcc, ' @ ', pathogenicity_threshold[k])
                string_to_write = '#' + gene.upper() + ' REVEL MCC: ' + str(revel_mcc) + ' @ ' + str(
                    pathogenicity_threshold[k]) + ' default' + '\n'
                file_to_write.write(string_to_write)
                default_revel = revel_mcc
                revel_mcc_l.append(default_revel)
            # print('matthews_corr_coef (REVEL) @ '+str(pathogenicity_threshold[k])+': ', revel_mcc)
            # print('Precision', prec_revel[k])
            # print('Recall', rec_revel[k])
            revel_MCC_at_diff_thresholds.append(revel_mcc)


            for d in range(len(revel_MCC_at_diff_thresholds) - 1):  # the optimum MCC score
                if d == 0:
                    revel_max_mcc += revel_MCC_at_diff_thresholds[d]
                else:
                    if revel_MCC_at_diff_thresholds[d + 1] > revel_max_mcc:
                        revel_max_mcc = revel_MCC_at_diff_thresholds[d + 1]
                        revel_max_mcc_threshold = pathogenicity_threshold[k]
                        # print('REVEL Precision @ ', revel_max_mcc_threshold, '=', counter_tp / (counter_tp + counter_fp))
                        # print('REVEL Recall @ ', revel_max_mcc_threshold, '=', counter_tp / (counter_tp + counter_fn))

        string_to_write = '#' + gene.upper() + ' REVEL MCC: ' + str(revel_max_mcc) + ' @ ' + str(
            revel_max_mcc_threshold) + '\n'
        file_to_write.write(string_to_write)

        revel_thresholds_identified_using_training_data.append(revel_max_mcc_threshold)
        revel_mcc_identified_for_all_genes.append(revel_max_mcc)

        vest4_MCC_at_diff_thresholds = []
        vest4_max_mcc = 0
        vest4_max_mcc_threshold = 0
        prec_vest4 = []
        rec_vest4 = []

        for k in range(len(pathogenicity_threshold)):
            vest4_pred = []
            for i in range(len(revel_scores)):
                if float(vest4_predictions[i]) >= pathogenicity_threshold[k]:
                    i = 1
                    vest4_pred.append(i)
                elif float(vest4_predictions[i]) < pathogenicity_threshold[k]:
                    i = 0
                    vest4_pred.append(i)
            counter_tp = 0
            counter_fn = 0
            counter_fp = 0
            counter_tn = 0
            for m in range(len(vest4_predicted_pathogenic)):
                if vest4_predicted_pathogenic[m] >= pathogenicity_threshold[k]:
                    counter_tp += 1
                elif vest4_predicted_pathogenic[m] < pathogenicity_threshold[k]:
                    counter_fn += 1
            for l in range(len(vest4_predicted_nonpathogenic)):
                if vest4_predicted_nonpathogenic[l] >= pathogenicity_threshold[k]:
                    counter_fp += 1
                    # print('counter_fp: ', counter_fp)
                elif vest4_predicted_nonpathogenic[l] < pathogenicity_threshold[k]:
                    counter_tn += 1
            prec_vest4.append(
                counter_tp / (counter_tp + counter_fp) if counter_tp + counter_fp != 0 and counter_tp != 0 else 0)
            rec_vest4.append(counter_tp / (counter_tp + counter_fn) if counter_tp + counter_fn != 0 else 0)
            accuracy = (counter_tp + counter_tn) / (counter_tp + counter_tn + counter_fp + counter_fn)
            vest4_mcc = matthews_corrcoef(binary_class, vest4_pred)
            # vest4_mcc_list.append(vest4_mcc)
            # print('matthews_corr_coef (VEST4) @ '+str(pathogenicity_threshold[k])+': ', vest4_mcc)
            if pathogenicity_threshold[k] == 0.5:
                # print('VEST4 Precision @ ', pathogenicity_threshold[k], '=', prec_vest4[k])
                # print('VEST4 Recall @ ', pathogenicity_threshold[k], '=', rec_vest4[k])
                string_to_write = '#' + gene.upper() + ' VEST4 MCC: ' + str(vest4_mcc) + ' @ ' + str(
                    pathogenicity_threshold[k]) + '\n'
                file_to_write.write(string_to_write)
                default_vest4 = vest4_mcc
                vest4_mcc_l.append(default_vest4)
            vest4_MCC_at_diff_thresholds.append(vest4_mcc)

            for d in range(len(vest4_MCC_at_diff_thresholds) - 1):  # the optimum MCC score
                if d == 0:
                    vest4_max_mcc += vest4_MCC_at_diff_thresholds[d]
                else:
                    if vest4_MCC_at_diff_thresholds[d + 1] > vest4_max_mcc:
                        vest4_max_mcc = vest4_MCC_at_diff_thresholds[d + 1]
                        vest4_max_mcc_threshold = pathogenicity_threshold[k]
                        # print('VEST4 Precision @ ', vest4_max_mcc_threshold, '=', counter_tp / (counter_tp + counter_fp))
                        # print('VEST4 Recall @ ', vest4_max_mcc_threshold, '=', counter_tp / (counter_tp + counter_fn))

        if transcript_found == False:
            print('VEST4 prediction NOT found for Transcript of interest!!')
        string_to_write = '#' + gene.upper() + ' VEST4 MCC: ' + str(vest4_max_mcc) + ' @ ' + str(
            vest4_max_mcc_threshold) + '\n'
        file_to_write.write(string_to_write)
        vest4_thresholds_identified_using_training_data.append(vest4_max_mcc_threshold)

        if transcript_found == True:
            vest4_mcc_identified_for_all_genes.append(vest4_max_mcc)
        elif transcript_found == False:
            vest4_mcc_identified_for_all_genes.append('-')
        #############################################
        ##########################################
        '''Validating the threshold identified using the training data'''

        # for i in genes:
        # gene = i
        # print('\n*************************   Validation  ' + gene.upper())
        string_to_write = '\n*************************   Validation  ' + gene.upper() + '\n'
        file_to_write.write(string_to_write)
        # pathogenicity_threshold_numbers = 51
        # revel_pathogenicity_threshold = 0
        # vest4_pathogenicity_threshold = 0
        # for k in range(len(revel_thresholds)):
        #     if gene_list[k] == gene:
        #         revel_pathogenicity_threshold += revel_thresholds[k]
        #         vest4_pathogenicity_threshold += vest4_thresholds[k]
        # test_set = pd.read_csv('/Users/mdefsss2/x_linked_genes/' + gene + '/test_set.csv')

        var_class = test_set['class'].tolist()
        classes = test_set['class_binary'].tolist()
        ensembl_transcripts = test_set['Ensembl_transcriptid'].tolist()
        coordinates = test_set['pos(1-based)'].tolist()
        revel_scores = test_set['REVEL_score'].tolist()
        vest4_scores = test_set['VEST4_score'].tolist()
        # transcript = ''
        transcript_found = True  # the predictions for the variant in transcript og interest not always available
        # for i in range(len(genes_transcripts)):  # obtaining the trascript used
        #     if genes_transcripts['gene'][i] == gene:
        #         transcript += genes_transcripts['transcript'][i].strip()
        # print('transcript retrieved: ', transcript)
        vest4_predictions = []
        # pph2_predictions = []
        for i in range(len(ensembl_transcripts)):
            strings = str(ensembl_transcripts[i])
            if ';' in strings:
                num_of_predictions = ensembl_transcripts[i].split(';')
                vest4_pred = vest4_scores[i].split(';')
                for k in range(len(num_of_predictions)):
                    if num_of_predictions[k] == transcript:
                        try:
                            vest4_predictions.append(float(vest4_pred[k]))
                        except ValueError as e:
                            # print(e) # the transcript of interest wasnt available & the first transcript prediction is therefore chosen instead
                            list_index_to_drop.append(i)
                    # elif transcript not in num_of_predictions:
                    # print('Transcript of inerest MISSING!! at: ', coordinates[k])
            elif strings.strip() == transcript.strip():
                vest4_predictions.append(float(vest4_scores[i]))
            # else:
            #     vest4_predictions = vest4_scores
        ########################################
        '''improving MCC scores for VEST4 in cases where there are multiple predictions for multiplr transcripts, where sorting by scores in EXCEL doesn't do the job.
        Here, the scores for transcript of interest are written to a new file along with the classes so they can be sorted to identify a better pathogenicity threshold'''
        # np.savetxt('/Users/mdefsss2/dbNSFP4/'+gene+'_vest_scores.csv', [p for p in zip(classes, vest4_predictions)], delimiter=',', fmt='%s')
        ########################################
        revel_predicted_pathogenic = []
        revel_predicted_nonpathogenic = []
        vest4_predicted_pathogenic = []
        vest4_predicted_nonpathogenic = []

        list_index_to_drop.sort(reverse=True)
        if list_index_to_drop:
            # print(f"Removing these index :{list_index_to_drop}")
            print(f"Going to remove {len(list_index_to_drop)} row(s) while doing validation!\n")
            for i in list_index_to_drop:
                del classes[i]
                del revel_scores[i]
            list_index_to_drop = []

        '''Mann Whitney U test'''
        for i in range(len(classes)):
            if classes[i] == 1:
                revel_predicted_pathogenic.append(revel_scores[i])
                vest4_predicted_pathogenic.append(vest4_predictions[i])
            elif classes[i] == 0:
                revel_predicted_nonpathogenic.append(revel_scores[i])
                vest4_predicted_nonpathogenic.append(vest4_predictions[i])

        file_to_write.write(f"number of pathogenic vars: {len(revel_predicted_pathogenic)}\n")
        file_to_write.write(f"number of non-pathogenic vars: {len(revel_predicted_nonpathogenic)}\n")
        # print('MannU test for REVEL: ', mannwhitneyu(revel_predicted_pathogenic, revel_predicted_nonpathogenic))
        # print('MannU test for VEST4: ', mannwhitneyu(vest4_predicted_pathogenic, vest4_predicted_nonpathogenic))
        #########################################################################
        '''for a number of pathogenicity thresholds, calculate the MCC scores for both tools'''
        binary_class = []
        for i in range(len(classes)):
            if classes[i] == 0 or classes[i] == 1:
                binary_class.append(classes[i])

        # revel_MCC_at_diff_thresholds = []
        revel_pred = []
        for i in range(len(revel_scores)):
            if float(revel_scores[i]) >= revel_max_mcc_threshold:
                i = 1
                revel_pred.append(i)
            elif float(revel_scores[i]) < revel_max_mcc_threshold:
                i = 0
                revel_pred.append(i)

        prec_revel = []
        rec_revel = []
        accuracy_revel = []
        original_prec_revel = []  # at 0.5 threshold
        original_rec_revel = []
        original_accuracy_revel = []
        for k in range(len(pathogenicity_threshold)):
            if pathogenicity_threshold[k] == revel_max_mcc_threshold:
                counter_tp = 0
                counter_fn = 0
                counter_fp = 0
                counter_tn = 0
                for m in range(len(revel_predicted_pathogenic)):
                    if float(revel_predicted_pathogenic[m]) >= revel_max_mcc_threshold:
                        counter_tp += 1
                    elif float(revel_predicted_pathogenic[m]) < revel_max_mcc_threshold:
                        counter_fn += 1
                for l in range(len(revel_predicted_nonpathogenic)):
                    if float(revel_predicted_nonpathogenic[l]) >= revel_max_mcc_threshold:
                        counter_fp += 1
                        # print('counter_fp: ', counter_fp)
                    elif float(revel_predicted_nonpathogenic[l]) < revel_max_mcc_threshold:
                        counter_tn += 1

                prec_revel.append(
                    counter_tp / (counter_tp + counter_fp) if counter_tp + counter_fp != 0 and counter_tp != 0 else 0)

                if counter_tp + counter_fn > 0:  # ZeroDivisionError
                    rec_revel.append(counter_tp / (counter_tp + counter_fn))
                else:
                    rec_revel.append(0)

                accuracy = (counter_tp + counter_tn) / (counter_tp + counter_tn + counter_fp + counter_fn)
                accuracy_revel.append(round(accuracy, 2))
                # print('REVEL Precision @ ' + str(revel_max_mcc_threshold) + ': ', prec_revel)
                # print('REVEL Recall @ ' + str(revel_max_mcc_threshold) + ': ', rec_revel)
                # print('REVEL Accuracy @ ' + str(revel_max_mcc_threshold) + ': ', accuracy)

            # elif pathogenicity_threshold[k] == 0.5:
            #     counter_tp = 0
            #     counter_fn = 0
            #     counter_fp = 0
            #     counter_tn = 0
            #     for m in range(len(revel_predicted_pathogenic)):
            #         if revel_predicted_pathogenic[m] >= revel_max_mcc_threshold:
            #             counter_tp += 1
            #         elif revel_predicted_pathogenic[m] < revel_max_mcc_threshold:
            #             counter_fn += 1
            #     for l in range(len(revel_predicted_nonpathogenic)):
            #         if revel_predicted_nonpathogenic[l] >= revel_max_mcc_threshold:
            #             counter_fp += 1
            #             # print('counter_fp: ', counter_fp)
            #         elif revel_predicted_nonpathogenic[l] < revel_max_mcc_threshold:
            #             counter_tn += 1
            #     original_prec_revel.append(counter_tp / (counter_tp + counter_fp) if counter_tp + counter_fp != 0 and counter_tp != 0 else 0)
            #     original_rec_revel.append(counter_tp / (counter_tp + counter_fn))
            #     accuracy = (counter_tp + counter_tn) / (counter_tp + counter_tn + counter_fp + counter_fn)
            #     original_accuracy_revel.append(round(accuracy, 2))
            # print('REVEL Precision @ ' + str(revel_max_mcc_threshold) + ': ', prec_revel)
            # print('REVEL Recall @ ' + str(revel_max_mcc_threshold) + ': ', rec_revel)
            # print('REVEL Accuracy @ ' + str(revel_max_mcc_threshold) + ': ', accuracy)
        revel_mcc = matthews_corrcoef(binary_class, revel_pred)  # validating the best identified threshold

        file_to_write.write(f"matthews_corr_coef (REVEL) @ {str(revel_max_mcc_threshold)} : {revel_mcc}\n")
        revel_mcc_at_identified_threshold.append(revel_mcc)
        revel_MCC_at_diff_thresholds.append(revel_mcc)
        revel_precision_tested_at_identified_threshold.append(prec_revel)
        revel_recall_tested_at_identified_threshold.append(rec_revel)
        revel_accuracy_tested_at_identified_threshold.append(accuracy_revel)
        # revel_precision_tested_at_original_threshold.append(original_prec_revel)
        # revel_recall_tested_at_original_threshold.append(original_rec_revel)
        # revel_accuracy_tested_at_original_threshold.append(original_accuracy_revel)

        # vest4_MCC_at_diff_thresholds = []
        vest4_pred = []
        for i in range(len(vest4_predictions)):
            # if vest4_predictions[i] == float:
            if float(vest4_predictions[i]) >= vest4_max_mcc_threshold:
                i = 1
                vest4_pred.append(i)
            elif float(vest4_predictions[i]) < vest4_max_mcc_threshold:
                i = 0
                vest4_pred.append(i)
        prec_vest4 = []
        rec_vest4 = []
        accuracy_vest4 = []
        for k in range(len(pathogenicity_threshold)):
            if pathogenicity_threshold[k] == vest4_max_mcc_threshold:
                counter_tp = 0
                counter_fn = 0
                counter_fp = 0
                counter_tn = 0
                for m in range(len(vest4_predicted_pathogenic)):
                    if vest4_predicted_pathogenic[m] >= vest4_max_mcc_threshold:
                        counter_tp += 1
                    elif vest4_predicted_pathogenic[m] < vest4_max_mcc_threshold:
                        counter_fn += 1
                for l in range(len(vest4_predicted_nonpathogenic)):
                    if vest4_predicted_nonpathogenic[l] >= vest4_max_mcc_threshold:
                        counter_fp += 1
                        # print('counter_fp: ', counter_fp)
                    elif vest4_predicted_nonpathogenic[l] < vest4_max_mcc_threshold:
                        counter_tn += 1
                prec_vest4.append(
                    counter_tp / (counter_tp + counter_fp) if counter_tp + counter_fp != 0 and counter_tp != 0 else 0)

                if counter_tp + counter_fn > 0:  # ZeroDivisionError
                    rec_vest4.append(counter_tp / (counter_tp + counter_fn))
                else:
                    rec_vest4.append(0)

                accuracy = (counter_tp + counter_tn) / (counter_tp + counter_tn + counter_fp + counter_fn)
                accuracy_vest4.append(round(accuracy, 2))
                file_to_write.write(f"vest4 Precision @ {str(vest4_max_mcc_threshold)} : {prec_vest4}\n")
                file_to_write.write(f"vest4 Recall @ {str(vest4_max_mcc_threshold)} : {rec_vest4}\n")
                file_to_write.write(f"vest4 Accuracy @ {str(vest4_max_mcc_threshold)} : {accuracy}\n")

        vest4_mcc = matthews_corrcoef(binary_class, vest4_pred)  # validating the best identified threshold
        file_to_write.write(f"matthews_corr_coef (VEST4) @ {str(vest4_max_mcc_threshold)} : {vest4_mcc}\n")
        # vest4_mcc_at_identified_threshold.append(vest4_mcc)
        vest4_precision_tested_at_identified_threshold.append(prec_vest4)
        vest4_recall_tested_at_identified_threshold.append(rec_vest4)
        vest4_accuracy_tested_at_identified_threshold.append(accuracy_vest4)

        if transcript_found == True:
            vest4_mcc_at_identified_threshold.append(vest4_mcc)
        elif transcript_found == False:
            pass
            # vest4_mcc_at_identified_threshold.append('-')

        # vest4_MCC_at_diff_thresholds.append(vest4_mcc)

        # SMALL_SIZE = 12
        # MEDIUM_SIZE = 14
        # BIGGER_SIZE = 16
        # plt.rc('font', size=SMALL_SIZE)
        # plt.rc('axes', titlesize=SMALL_SIZE)
        # plt.rc('axes', labelsize=MEDIUM_SIZE)
        # plt.rc('xtick', labelsize=SMALL_SIZE)
        # plt.rc('ytick', labelsize=SMALL_SIZE)
        # plt.rc('legend', fontsize=SMALL_SIZE)
        # plt.rc('figure', titlesize=BIGGER_SIZE)
        # mydensity = 1
        # myalpha = 0.8
        #############################################
        '''if plotting all genes' MCC scores for each tool on one plot'''
        # fig, ax = plt.subplots()
        #######################################################
        '''comment out if plotting all genes' MCC on one graph for a tool'''
        # fig = plt.figure()

        #######################################################
        '''plotting VEST4 & REVEL MCC scores for each gene for different thresholds on one plot'''
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(pathogenicity_threshold, revel_MCC_at_diff_thresholds, label='REVEL')
        # # ax.plot(pathogenicity_threshold, vest4_MCC_at_diff_thresholds, label='VEST4')
        # ax.set_xlabel('Thresholds')
        # ax.set_ylabel('MCC')
        # ax.set_title(gene)
        # plt.legend()
        # plt.show()
        #######################################################
        '''plotting each tool's MCC scores for all genes together on one plot'''
        #     # plt.plot(pathogenicity_threshold, revel_MCC_at_diff_thresholds, label='')
        #     ax.plot(pathogenicity_threshold, vest4_MCC_at_diff_thresholds, label='')
        #     plt.xlabel('Thresholds')
        #     plt.ylabel('MCC')
        #     plt.title('VEST4')
        #     plt.legend()
        # plt.show()
        #######################################################
        '''showing the best MCC scores for each gene in a scatter plot'''

        # plt.close('all')
        #######################################################
        # print('####################################################')
        # print('REVEL MCCc for each gene identified using optimum Thresholds from training data :\n')
        # for i in range(len(revel_thresholds)):
        #     print(revel_thresholds[i], round(revel_mcc_at_identified_thresholds[i], 2))
        #
        # print('####################################################')
        # print('VEST4 MCCc for each gene identified using optimum Thresholds from training data :\n')
        # for i in range(len(vest4_thresholds)):
        #     if vest4_thresholds[i] == float or vest4_thresholds[i] == int:
        #         print(vest4_thresholds[i], round(vest4_mcc_at_identified_thresholds[i], 2))
        #     else:
        #         print(vest4_thresholds[i], vest4_mcc_at_identified_thresholds[i])

        # SMALL_SIZE = 12
        # MEDIUM_SIZE = 14
        # BIGGER_SIZE = 16
        # plt.rc('font', size=SMALL_SIZE)
        # plt.rc('axes', titlesize=SMALL_SIZE)
        # plt.rc('axes', labelsize=MEDIUM_SIZE)
        # plt.rc('xtick', labelsize=SMALL_SIZE)
        # plt.rc('ytick', labelsize=SMALL_SIZE)
        # plt.rc('legend', fontsize=SMALL_SIZE)
        # plt.rc('figure', titlesize=BIGGER_SIZE)
        # mydensity = 1
        # myalpha = 0.8
        #############################################
        '''if plotting all genes' MCC scores for each tool on one plot'''
        # fig, ax = plt.subplots()
        #######################################################
        '''comment out if plotting all genes' MCC on one graph for a tool'''
        # fig = plt.figure()

        #######################################################
        '''plotting VEST4 & REVEL MCC scores for each gene for different thresholds on one plot'''
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(pathogenicity_threshold, revel_MCC_at_diff_thresholds, label='REVEL')
        # # ax.plot(pathogenicity_threshold, vest4_MCC_at_diff_thresholds, label='VEST4')
        # ax.set_xlabel('Thresholds')
        # ax.set_ylabel('MCC')
        # ax.set_title(gene)
        # plt.legend()
        # plt.show()
        #######################################################
        '''plotting each tool's MCC scores for all genes together on one plot'''
    #     # plt.plot(pathogenicity_threshold, revel_MCC_at_diff_thresholds, label='')
    #     ax.plot(pathogenicity_threshold, vest4_MCC_at_diff_thresholds, label='')
    #     plt.xlabel('Thresholds')
    #     plt.ylabel('MCC')
    #     plt.title('VEST4')
    #     plt.legend()
    if not break_indicator:
        file_to_write.write(f"\n*************************     {gene.upper()}\n")

        mean_revel_threshold_identified_using_training_data.append(
            np.mean(revel_thresholds_identified_using_training_data))  # mean of the 10 replicated thresholds
        mean_vest4_threshold_identified_using_training_data.append(
            np.mean(vest4_thresholds_identified_using_training_data))  # mean of the 10 replicated thresholds
        stdev_revel_threshold_identified_using_training_data.append(
            np.std(revel_thresholds_identified_using_training_data))  # mean of the 10 replicated thresholds
        stdev_vest4_threshold_identified_using_training_data.append(
            np.std(vest4_thresholds_identified_using_training_data))  # mean of the 10 replicated thresholds

        file_to_write.write(
            f"{gene} 10 REVEL Thresholds identified using training data: {revel_thresholds_identified_using_training_data}\n")
        file_to_write.write(
            f"{gene} 10 VEST4 Thresholds identified using training data: {vest4_thresholds_identified_using_training_data}\n")

        revel_mcc_mean = np.mean(revel_mcc_at_identified_threshold)
        revel_mcc_mean_for_all_genes.append(revel_mcc_mean)
        revel_mcc_stDev = np.std(revel_mcc_at_identified_threshold)
        revel_mcc_stdev_for_all_genes.append(revel_mcc_stDev)
        vest4_mcc_mean = np.mean(vest4_mcc_at_identified_threshold)
        vest4_mcc_mean_for_all_genes.append(vest4_mcc_mean)
        vest4_mcc_stDev = np.std(vest4_mcc_at_identified_threshold)
        vest4_mcc_stdev_for_all_genes.append(vest4_mcc_stDev)

        revel_precision_tested_at_identified_threshold_mean = np.mean(revel_precision_tested_at_identified_threshold)
        revel_precision_tested_mean_for_all_genes.append(revel_precision_tested_at_identified_threshold_mean)

        revel_recall_tested_at_identified_threshold_mean = np.mean(revel_recall_tested_at_identified_threshold)
        revel_recall_tested_mean_for_all_genes.append(revel_recall_tested_at_identified_threshold_mean)

        revel_accuracy_tested_at_identified_threshold_mean = np.mean(revel_accuracy_tested_at_identified_threshold)
        revel_accuracy_tested_mean_for_all_genes.append(revel_accuracy_tested_at_identified_threshold_mean)

        vest4_precision_tested_at_identified_threshold_mean = np.mean(vest4_precision_tested_at_identified_threshold)
        vest4_precision_tested_mean_for_all_genes.append(vest4_precision_tested_at_identified_threshold_mean)

        vest4_recall_tested_at_identified_threshold_mean = np.mean(vest4_recall_tested_at_identified_threshold)
        vest4_recall_tested_mean_for_all_genes.append(vest4_recall_tested_at_identified_threshold_mean)

        vest4_accuracy_tested_at_identified_threshold_mean = np.mean(vest4_accuracy_tested_at_identified_threshold)
        vest4_accuracy_tested_mean_for_all_genes.append(vest4_accuracy_tested_at_identified_threshold_mean)

        revel_precision_tested_at_identified_threshold_std = np.std(revel_precision_tested_at_identified_threshold)
        revel_precision_tested_stdev_for_all_genes.append(revel_precision_tested_at_identified_threshold_std)

        revel_recall_tested_at_identified_threshold_std = np.std(revel_recall_tested_at_identified_threshold)
        revel_recall_tested_stdev_for_all_genes.append(revel_recall_tested_at_identified_threshold_std)

        revel_accuracy_tested_at_identified_threshold_std = np.std(revel_accuracy_tested_at_identified_threshold)
        revel_accuracy_tested_stdev_for_all_genes.append(revel_accuracy_tested_at_identified_threshold_std)

        vest4_precision_tested_at_identified_threshold_std = np.std(vest4_precision_tested_at_identified_threshold)
        vest4_precision_tested_stdev_for_all_genes.append(vest4_precision_tested_at_identified_threshold_std)

        vest4_recall_tested_at_identified_threshold_std = np.std(vest4_recall_tested_at_identified_threshold)
        vest4_recall_tested_stdev_for_all_genes.append(vest4_recall_tested_at_identified_threshold_std)

        vest4_accuracy_tested_at_identified_threshold_std = np.std(vest4_accuracy_tested_at_identified_threshold)
        vest4_accuracy_tested_stdev_for_all_genes.append(vest4_accuracy_tested_at_identified_threshold_std)

        '''metrics @ the original 0.5 threshold'''
        # revel_precision_tested_at_original_threshold_mean = np.mean(revel_precision_tested_at_original_threshold)
        # original_revel_precision_tested_mean_for_all_genes.append(revel_precision_tested_at_original_threshold_mean)
        #
        # revel_recall_tested_at_original_threshold_mean = np.mean(revel_recall_tested_at_original_threshold)
        # original_revel_recall_tested_mean_for_all_genes.append(revel_recall_tested_at_original_threshold_mean)
        #
        # revel_accuracy_tested_at_original_threshold_mean = np.mean(revel_accuracy_tested_at_original_threshold)
        # original_revel_accuracy_tested_mean_for_all_genes.append(revel_accuracy_tested_at_original_threshold_mean)
        #
        # revel_precision_tested_at_original_threshold_std = np.std(revel_precision_tested_at_original_threshold)
        # original_revel_precision_tested_stdev_for_all_genes.append(revel_precision_tested_at_original_threshold_std)
        #
        # revel_recall_tested_at_original_threshold_std = np.std(revel_recall_tested_at_original_threshold)
        # original_revel_recall_tested_stdev_for_all_genes.append(revel_recall_tested_at_original_threshold_std)
        #
        # revel_accuracy_tested_at_original_threshold_std = np.std(revel_accuracy_tested_at_original_threshold)
        # original_revel_accuracy_tested_stdev_for_all_genes.append(revel_accuracy_tested_at_original_threshold_std)

        file_to_write.write(f"{gene} :REVEL MCC mean & stdev: {revel_mcc_mean} {revel_mcc_stDev}\n")
        file_to_write.write(f"{gene} :VEST4 MCC mean & stdev: {vest4_mcc_mean} {vest4_mcc_stDev}\n")

        mean_revel = np.mean(revel_mcc_l)
        std_revel = np.std(revel_mcc_l)
        mean_vest4 = np.mean(vest4_mcc_l)
        std_vest4 = np.std(vest4_mcc_l)

        default_threshold = 0.5
        results_file.write(
            f"{gene}\t{occurance_0}\t{occurance_1}\t{default_threshold}\t{mean_revel:.2f}\t{default_threshold}\t{mean_vest4:.2f}\t{np.mean(revel_thresholds_identified_using_training_data):.2f}\t{revel_mcc_mean:.2f}±{revel_mcc_stDev:.2f}\t{np.mean(vest4_thresholds_identified_using_training_data):.2f}\t{vest4_mcc_mean:.2f}±{vest4_mcc_stDev:.2f}\n")

        mean_and_std_file_default.write(f'{gene}\t{occurance_0}\t{occurance_1}\t{mean_revel}\t{std_revel}\t{mean_vest4}\t{std_vest4}\n')
        mean_and_std_file_optimized.write(f'{gene}\t{occurance_0}\t{occurance_1}\t{revel_mcc_mean}\t{revel_mcc_stDev}\t{vest4_mcc_mean}\t{vest4_mcc_stDev}\n')
    file_to_write.close()
results_file.close()
mean_and_std_file_default.close()
mean_and_std_file_optimized.close()

print(f"bad_files_first: {bad_files_first}\n {len(bad_files_first)}")
print(f"bad_files_second: {bad_files_second}\n {len(bad_files_second)}")

print('\nThe Genes\' mean & stDev REVEL Thresholds identified using training data: ')
for k in range(len(mean_revel_threshold_identified_using_training_data)):
    print(genes[k].upper(), mean_revel_threshold_identified_using_training_data[k],
          stdev_revel_threshold_identified_using_training_data[k])
print('\nThe Genes\' mean & stDev VEST4 Thresholds identified using training data: ')
for k in range(len(mean_vest4_threshold_identified_using_training_data)):
    print(genes[k].upper(), mean_vest4_threshold_identified_using_training_data[k],
          stdev_vest4_threshold_identified_using_training_data[k])
# print('\nThe Genes\' REVEL MCC means & stDev obtained using validation data: ')
# for k in range(len(revel_mcc_stdev_for_all_genes)):
#     print(genes[k].upper(), revel_mcc_mean_for_all_genes[k], revel_mcc_stdev_for_all_genes[k])
# print('\nThe Genes\' VEST4 MCC means & stDev obtained using validation data: ')
# for k in range(len(vest4_mcc_stdev_for_all_genes)):
#     print(genes[k].upper(), vest4_mcc_mean_for_all_genes[k], vest4_mcc_stdev_for_all_genes[k])
# print('\nThe Genes\' REVEL Precision means & stDev obtained using validation data: ')
# for k in range(len(revel_precision_tested_mean_for_all_genes)):
#     print(genes[k].upper(), revel_precision_tested_mean_for_all_genes[k], revel_precision_tested_stdev_for_all_genes[k])
# print('\nThe Genes\' VEST4 Precision means & stDev obtained using validation data: ')
# for k in range(len(vest4_precision_tested_mean_for_all_genes)):
#     print(genes[k].upper(), vest4_precision_tested_mean_for_all_genes[k], vest4_precision_tested_stdev_for_all_genes[k])
#
# print('\nThe Genes\' REVEL recall means & stDev obtained using validation data: ')
# for k in range(len(revel_recall_tested_mean_for_all_genes)):
#     print(genes[k].upper(), revel_recall_tested_mean_for_all_genes[k], revel_recall_tested_stdev_for_all_genes[k])
# print('\nThe Genes\' VEST4 recall means & stDev obtained using validation data: ')
# for k in range(len(vest4_recall_tested_mean_for_all_genes)):
#     print(genes[k].upper(), vest4_recall_tested_mean_for_all_genes[k], vest4_recall_tested_stdev_for_all_genes[k])
#
# print('\nThe Genes\' REVEL Accuracy means & stDev obtained using validation data: ')
# for k in range(len(revel_accuracy_tested_mean_for_all_genes)):
#     print(genes[k].upper(), revel_accuracy_tested_mean_for_all_genes[k], revel_accuracy_tested_stdev_for_all_genes[k])
# print('\nThe Genes\' VEST4 Accuracy means & stDev obtained using validation data: ')
# for k in range(len(vest4_accuracy_tested_mean_for_all_genes)):
#     print(genes[k].upper(), vest4_accuracy_tested_mean_for_all_genes[k], vest4_accuracy_tested_stdev_for_all_genes[k])








# print('\nThe Genes\' REVEL Precision means & stDev obtained using validation data at 0.5 threshold: ')
# for k in range(len(original_revel_precision_tested_mean_for_all_genes)):
#     print(genes[k].upper(), original_revel_precision_tested_mean_for_all_genes[k], original_revel_precision_tested_stdev_for_all_genes[k])
# print('\nThe Genes\' REVEL recall means & stDev obtained using validation data at 0.5 threshold: ')
# for k in range(len(original_revel_recall_tested_mean_for_all_genes)):
#     print(genes[k].upper(), original_revel_recall_tested_mean_for_all_genes[k], original_revel_recall_tested_stdev_for_all_genes[k])
# print('\nThe Genes\' REVEL Accuracy means & stDev obtained using validation data at 0.5 threshold: ')
# for k in range(len(original_revel_accuracy_tested_mean_for_all_genes)):
#     print(genes[k].upper(), original_revel_accuracy_tested_mean_for_all_genes[k], original_revel_accuracy_tested_stdev_for_all_genes[k])

# plt.show()
#######################################################
'''showing the best MCC scores for each gene in a scatter plot'''

# plt.close('all')
# print('####################################################')
# print('REVEL Thresholds & MCCc for each gene:\n')
# for i in range(len(revel_thresholds_identified_for_genes)):
#     print(revel_thresholds_identified_for_genes[i], round(revel_mcc_identified_for_genes[i], 2))
#
# print('####################################################')
# print('VEST4 Thresholds & MCCs for each gene:\n')
# for i in range(len(vest4_thresholds_identified_for_genes)):
#     if vest4_mcc_identified_for_genes[i] == float or vest4_mcc_identified_for_genes[i] == int:
#         print(vest4_thresholds_identified_for_genes[i], round(vest4_mcc_identified_for_genes[i], 2))
#     else:
#         print(vest4_thresholds_identified_for_genes[i], vest4_mcc_identified_for_genes[i])
