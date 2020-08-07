#Batch process'_g.csv' and'_validator.txt' files, and output txt files that meet the requirements of dbnsFP tools
import glob
import pandas as pd

def process_file(in_file_name, out_file_name, column_name_list, indicator):
    if indicator == 1:
        df = pd.read_csv(in_file_name, comment="#", sep="\t")
        df = df.dropna(subset = ["GRCh37_POS"])
    else:
        df = pd.read_csv(in_file_name, comment="#")
        df = df.dropna(subset = ["Position"])
    df.to_csv(out_file_name, index=False, header=False, float_format="%.f", columns=column_name_list, sep=" ")

def main():
    validator_filename_list = glob.glob("*_validator.txt") 
    #This file is obtained by processing the cDNA file at https://variantvalidator.org/service/validate/batch/
    _g_filename_list = glob.glob("*_g.csv")
    #This file was obtained by searching for genes at https://gnomad.broadinstitute.org/
    validator_columns_list = ["GRCh37_CHR", "GRCh37_POS", "GRCh37_REF", "GRCh37_ALT"]
    _g_columns_list = ["Chromosome", "Position", "Reference", "Alternate"]
    for validator_filename in validator_filename_list:
        gene_name = validator_filename.split("_")[0]
        out_file_name = gene_name + "_HGMD.txt"
        process_file(validator_filename, out_file_name, validator_columns_list, 1)
    for _g in _g_filename_list:
        gene_name = _g.split("_")[0]
        out_file_name = gene_name + "_gnomAD.txt"
        process_file(_g, out_file_name, _g_columns_list, 0)

if __name__ == "__main__":
    main()
