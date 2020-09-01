import matplotlib.pyplot as plt
import pandas as pd
import sys

x_data = []
y_data = []

def to_plot(name_to_save, column_name1, column_name2, data):
    try:
        mean= data[column_name1]
        std= data[column_name2]
        genes = data['gene']
    except KeyError:
        print("Column name not found!")
        sys.exit()
    plt.figure(figsize=(50,5))
    plt.figure(1).subplots_adjust(**dict(left=0.1, right=.8, bottom=.175, top=.9, wspace=.1, hspace=.1))
    plt.errorbar(genes, mean, linestyle='None', fmt='-o')
    plt.xticks(rotation=-90)
    plt.savefig(name_to_save + '.png')
    # plt.show()

def main():
    filename = "mean_and_std_optimized.tsv"
    try:
        data = pd.read_csv(filename, sep='\t', index_col=None)
    except FileNotFoundError:
        print("File to store data is not found!\n")
        sys.exit()
    to_plot("optimized_revel", "mean revel", "std revel", data)
    to_plot("optimized_vest4", "mean vest4", "std vest4", data)
    # break

if __name__ == '__main__':
    main()
